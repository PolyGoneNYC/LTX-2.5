"""Re-voice a generated video's speaking segments for one enrolled character.

Pipeline: run Light-ASD to find who's talking when -> match each face track's identity via
InsightFace against an enrolled character profile -> for the track that matches, convert its
speaking segments' audio to the character's reference voice via OpenVoice -> splice back in.

See README.md for install/usage and for an honest note on accuracy: this chains three
independent ML models and has not been run end-to-end (no GPU/test video in the environment this
was built in). Verify the printed match scores and segment timestamps on your first few runs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

FPS_ASSUMED_BY_LIGHT_ASD = 25  # Light-ASD's own pipeline re-samples video to 25fps internally.


# ---------------------------------------------------------------------------
# Stage 1: run Light-ASD, load its tracks/scores
# ---------------------------------------------------------------------------


def run_light_asd(video_path: str, light_asd_repo: str, weight_path: str, work_dir: Path) -> tuple[list, list]:
    """Invoke Light-ASD's own Columbia_test.py as a subprocess (matching its documented usage),
    then load the tracks.pckl / scores.pckl it writes.

    Returns (tracks, scores) -- see module docstring for their exact structure.
    """
    video_folder = work_dir / "light_asd_input"
    video_folder.mkdir(parents=True, exist_ok=True)
    video_name = "clip"
    dest = video_folder / f"{video_name}{Path(video_path).suffix}"
    if not dest.exists():
        dest.symlink_to(Path(video_path).resolve())

    cmd = [
        "python",
        "Columbia_test.py",
        "--videoName",
        video_name,
        "--videoFolder",
        str(video_folder),
        "--pretrainModel",
        weight_path,
    ]
    subprocess.run(cmd, cwd=light_asd_repo, check=True)

    pywork = video_folder / video_name / "pywork"
    with open(pywork / "tracks.pckl", "rb") as f:
        tracks = pickle.load(f)
    with open(pywork / "scores.pckl", "rb") as f:
        scores = pickle.load(f)
    return tracks, scores


# ---------------------------------------------------------------------------
# Stage 2: identify which track is the enrolled character
# ---------------------------------------------------------------------------


def crop_face_at_frame(
    video_path: str, frame_idx: int, track: dict, proc_frame_idx: int, crop_scale: float = 0.40
) -> np.ndarray:
    """Reproduce Light-ASD's own crop_video() padding-crop math for one frame, so the crop we
    feed to InsightFace matches what Light-ASD itself considered "this track's face".

    frame_idx is a frame number in Light-ASD's own 25fps-resampled space (it always re-encodes
    the input to 25fps before tracking), which generally does NOT match video_path's native frame
    rate -- so we seek by timestamp (frame_idx / 25s) rather than by raw frame count.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, (frame_idx / FPS_ASSUMED_BY_LIGHT_ASD) * 1000)
    ok, image = cap.read()
    cap.release()
    if not ok:
        t = frame_idx / FPS_ASSUMED_BY_LIGHT_ASD
        raise ValueError(f"Could not read frame {frame_idx} (t={t:.3f}s) from {video_path}")

    proc = track["proc_track"]
    bs = proc["s"][proc_frame_idx]
    bsi = int(bs * (1 + 2 * crop_scale))
    padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
    my = proc["y"][proc_frame_idx] + bsi
    mx = proc["x"][proc_frame_idx] + bsi
    y0, y1 = int(my - bs), int(my + bs * (1 + 2 * crop_scale))
    x0, x1 = int(mx - bs * (1 + crop_scale)), int(mx + bs * (1 + crop_scale))
    face = padded[y0:y1, x0:x1]
    return face


def identify_character_track(
    video_path: str,
    tracks: list,
    face_app,
    target_embedding: np.ndarray,
    match_threshold: float,
) -> tuple[int | None, float]:
    """For each face track, sample a few representative frames, embed them with InsightFace, and
    compare (cosine similarity) against the enrolled character's embedding. Returns
    (best_track_idx, best_score), or (None, best_score) if nothing clears match_threshold.
    """
    best_idx, best_score = None, -1.0
    for track_idx, track in enumerate(tracks):
        frame_ids = track["track"]["frame"]
        # Sample up to 5 frames spread across the track rather than just the first, so a blurry
        # or turned-away opening frame doesn't sink an otherwise-good match.
        sample_positions = np.linspace(0, len(frame_ids) - 1, num=min(5, len(frame_ids)), dtype=int)
        similarities = []
        for pos in sample_positions:
            frame_idx = frame_ids[pos]
            crop = crop_face_at_frame(video_path, frame_idx, track, pos)
            if crop.size == 0:
                continue
            faces = face_app.get(crop)
            if not faces:
                continue
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            similarities.append(float(np.dot(largest.normed_embedding, target_embedding)))
        if not similarities:
            continue
        track_score = float(np.median(similarities))
        print(f"  track {track_idx}: identity similarity = {track_score:.3f} ({len(similarities)} frames sampled)")
        if track_score > best_score:
            best_idx, best_score = track_idx, track_score

    if best_score < match_threshold:
        return None, best_score
    return best_idx, best_score


# ---------------------------------------------------------------------------
# Stage 3: turn a track's per-frame speaking scores into time segments
# ---------------------------------------------------------------------------


def speaking_segments(
    track: dict,
    track_scores: list[float],
    speaking_threshold: float,
    min_segment_seconds: float,
    fps: float = FPS_ASSUMED_BY_LIGHT_ASD,
) -> list[tuple[float, float]]:
    """Threshold the track's per-frame ASD scores into contiguous (start_sec, end_sec) segments,
    dropping segments shorter than min_segment_seconds (avoids flickering on single noisy frames).
    """
    frame_ids = track["track"]["frame"]
    is_speaking = [s > speaking_threshold for s in track_scores]

    segments = []
    seg_start = None
    for i, speaking in enumerate(is_speaking):
        if speaking and seg_start is None:
            seg_start = frame_ids[i]
        elif not speaking and seg_start is not None:
            seg_end = frame_ids[i - 1] + 1
            segments.append((seg_start, seg_end))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, frame_ids[-1] + 1))

    return [(start / fps, end / fps) for start, end in segments if (end - start) / fps >= min_segment_seconds]


# ---------------------------------------------------------------------------
# Stage 4: re-voice the identified segments and splice them back in
# ---------------------------------------------------------------------------


def extract_audio(video_path: str, out_wav: Path) -> None:
    """Extract audio at its native sample rate and channel count -- no forced -ac/-ar, so
    revoice_and_splice can preserve the original's channel count and quality everywhere except
    the segments actually being re-voiced.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", str(out_wav)],
        check=True,
        capture_output=True,
    )


def revoice_and_splice(
    video_path: str,
    segments: list[tuple[float, float]],
    converter,
    target_se,
    work_dir: Path,
    tau: float,
    crossfade_seconds: float = 0.08,
) -> Path:
    """For each segment, run OpenVoice tone-color conversion (source style extracted from that
    segment itself) and splice the converted audio back into a copy of the full track, with a
    short linear crossfade at each boundary to avoid audible clicks.

    Operates on the track's own native sample rate and channel count throughout; only the
    portion actually being re-voiced is downmixed to mono for OpenVoice (which requires it) and
    then duplicated back across channels for that segment alone -- audio outside the segments,
    and each channel's original stereo image there, is untouched. extract_se/convert internally
    resample whatever they're given to the model's own rate via librosa.load(sr=...), so segments
    are written at the track's native rate as-is; only the returned converted audio (written at
    the model's rate) needs resampling back.
    """
    import soundfile as sf
    import torch
    import torchaudio

    full_audio_path = work_dir / "full_audio.wav"
    extract_audio(video_path, full_audio_path)
    raw_audio, sample_rate = sf.read(full_audio_path, always_2d=True)  # raw_audio: [T, C]
    audio = raw_audio.T.astype(np.float64)  # [C, T]
    num_channels = audio.shape[0]

    crossfade_n = int(crossfade_seconds * sample_rate)

    for i, (start_s, end_s) in enumerate(segments):
        start_i, end_i = int(start_s * sample_rate), int(end_s * sample_rate)
        segment_mono = audio[:, start_i:end_i].mean(axis=0) if num_channels > 1 else audio[0, start_i:end_i]
        seg_path = work_dir / f"segment_{i}.wav"
        sf.write(seg_path, segment_mono, sample_rate)

        src_se = converter.extract_se([str(seg_path)], se_save_path=None)
        converted_path = work_dir / f"segment_{i}.converted.wav"
        converter.convert(
            audio_src_path=str(seg_path),
            src_se=src_se,
            tgt_se=target_se,
            output_path=str(converted_path),
            tau=tau,
        )
        converted, converted_sr = sf.read(converted_path)
        if converted_sr != sample_rate:
            converted = torchaudio.functional.resample(
                torch.from_numpy(converted.astype(np.float32)), converted_sr, sample_rate
            ).numpy()
        converted = converted.astype(np.float64)
        # OpenVoice's output length can differ slightly from the source segment's; trim/pad to fit
        # so splicing back doesn't shift everything after it.
        target_len = end_i - start_i
        if len(converted) > target_len:
            converted = converted[:target_len]
        elif len(converted) < target_len:
            converted = np.pad(converted, (0, target_len - len(converted)))

        n = min(crossfade_n, target_len // 2)
        if n > 0:
            fade_in = np.linspace(0, 1, n)
            fade_out = 1 - fade_in
            converted[:n] = converted[:n] * fade_in + segment_mono[:n] * fade_out
            converted[-n:] = converted[-n:] * fade_out + segment_mono[-n:] * fade_in

        audio[:, start_i:end_i] = np.tile(converted, (num_channels, 1))
        print(f"  re-voiced segment {start_s:.2f}s - {end_s:.2f}s")

    out_path = work_dir / "revoiced_audio.wav"
    sf.write(out_path, audio.T, sample_rate)  # soundfile expects [T, C] (frames-major), not [C, T]
    return out_path


def mux_video_with_audio(video_path: str, audio_path: Path, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-shortest",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--character-profile", required=True, help="Path to a profile JSON from enroll_character.py.")
    parser.add_argument("--light-asd-repo", required=True, help="Path to a checkout of Junhua-Liao/Light-ASD.")
    parser.add_argument("--light-asd-weight", required=True)
    parser.add_argument("--converter-config", required=True, help="OpenVoice ToneColorConverter config.json.")
    parser.add_argument("--converter-ckpt", required=True, help="OpenVoice ToneColorConverter checkpoint.")
    parser.add_argument("--output-video", required=True)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.35,
        help="Min cosine similarity to accept an identity match (default: 0.35 -- tune per README).",
    )
    parser.add_argument(
        "--speaking-threshold",
        type=float,
        default=0.0,
        help="Min Light-ASD score to count as 'speaking' (default: 0.0, its own positive/negative boundary).",
    )
    parser.add_argument("--min-segment-seconds", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.3, help="OpenVoice conversion strength.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Don't delete intermediate files (useful for inspecting matches/segments).",
    )
    args = parser.parse_args()

    profile = json.loads(Path(args.character_profile).read_text())
    target_embedding = np.array(profile["face_embedding"], dtype=np.float32)

    work_dir = Path(tempfile.mkdtemp(prefix="voice_lock_"))
    print(f"Working directory: {work_dir}")

    print("Stage 1/4: running Light-ASD (face tracking + active-speaker detection)...")
    tracks, scores = run_light_asd(args.input_video, args.light_asd_repo, args.light_asd_weight, work_dir)
    print(f"  found {len(tracks)} face track(s)")

    print(f"Stage 2/4: identifying which track is '{profile['name']}'...")
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    track_idx, match_score = identify_character_track(
        args.input_video, tracks, face_app, target_embedding, args.match_threshold
    )
    if track_idx is None:
        raise SystemExit(
            f"No face track matched '{profile['name']}' above --match-threshold {args.match_threshold} "
            f"(best similarity seen: {match_score:.3f}). Nothing to re-voice. Try lowering the "
            f"threshold, or check the reference photos / --keep-work-dir output."
        )
    print(f"  matched track {track_idx} (similarity {match_score:.3f})")

    print("Stage 3/4: finding speaking segments for that track...")
    segments = speaking_segments(
        tracks[track_idx], scores[track_idx], args.speaking_threshold, args.min_segment_seconds
    )
    if not segments:
        raise SystemExit(
            f"'{profile['name']}'s track never scored above --speaking-threshold "
            f"{args.speaking_threshold}. Nothing to re-voice."
        )
    print(f"  {len(segments)} segment(s): " + ", ".join(f"{s:.2f}-{e:.2f}s" for s, e in segments))

    print("Stage 4/4: re-voicing segments with OpenVoice and splicing...")
    from openvoice.api import ToneColorConverter

    converter = ToneColorConverter(args.converter_config, device=args.device)
    # Not passed as an enable_watermark=False constructor kwarg: ToneColorConverter.__init__
    # forwards **kwargs straight to OpenVoiceBaseClass.__init__(config_path, device=...), which
    # doesn't accept enable_watermark and would raise TypeError. Disabling it post-construction
    # instead skips add_watermark()'s effect (it no-ops when watermark_model is None) without
    # touching that constructor path.
    converter.watermark_model = None
    converter.load_ckpt(args.converter_ckpt)
    target_se = converter.extract_se([profile["reference_voice_path"]], se_save_path=None)

    revoiced_audio = revoice_and_splice(args.input_video, segments, converter, target_se, work_dir, args.tau)
    mux_video_with_audio(args.input_video, revoiced_audio, args.output_video)
    print(f"Done: {args.output_video}")

    if not args.keep_work_dir:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
