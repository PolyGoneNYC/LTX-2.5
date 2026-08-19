"""ComfyUI node: re-voice one enrolled character's speaking segments in a generated video.

Runs entirely inside the ComfyUI graph -- no separate script, no output file to hand back in.
Wire it in right after wherever your workflow produces a VIDEO (e.g. the LTX-2.5 subgraph's
output) and before SaveVideo.

This is a from-scratch ComfyUI port of the same pipeline as the standalone
voice_lock/lock_character_voice.py tool (same three libraries, same verified APIs), adapted to
work on ComfyUI's in-memory VIDEO/IMAGE/AUDIO types instead of file paths, so it can run as one
node in the graph instead of a terminal command after the fact.

Install: copy this whole `ltxv_voice_lock/` folder into ComfyUI/custom_nodes/, install its
dependencies INTO COMFYUI'S OWN PYTHON ENVIRONMENT (see README.md in this folder -- this is the
one real cost of running inside the ComfyUI process instead of an isolated venv: these are heavy
research-repo dependencies that can conflict with what ComfyUI itself needs), and restart ComfyUI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def _save_photo(image_tensor: torch.Tensor, out_path: Path) -> None:
    """Write a single ComfyUI IMAGE frame ([N,H,W,3] float 0-1, RGB) to a jpg file."""
    from PIL import Image

    frame = image_tensor[0]
    arr = (frame * 255).clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(arr, mode="RGB").save(out_path)


def _save_voice(audio: dict, out_path: Path) -> None:
    """Write a ComfyUI AUDIO ({'waveform': [B,C,T] float, 'sample_rate': int}) to a wav file."""
    import soundfile as sf

    waveform = audio["waveform"][0]  # [C, T]
    sf.write(out_path, waveform.T.cpu().numpy(), audio["sample_rate"])


# --- Pipeline stages, ported from voice_lock/lock_character_voice.py (see that file's more ---
# --- detailed comments; kept here as a self-contained copy so this folder installs standalone) ---

FPS_ASSUMED_BY_LIGHT_ASD = 25


def _run_light_asd(video_path: str, light_asd_repo: str, weight_path: str, work_dir: Path) -> tuple[list, list]:
    import pickle
    import subprocess

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
    result = subprocess.run(cmd, cwd=light_asd_repo, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"Light-ASD (Columbia_test.py) failed:\n{tail}")

    pywork = video_folder / video_name / "pywork"
    with open(pywork / "tracks.pckl", "rb") as f:
        tracks = pickle.load(f)
    with open(pywork / "scores.pckl", "rb") as f:
        scores = pickle.load(f)
    return tracks, scores


def _crop_face_at_frame(
    video_path: str, frame_idx: int, track: dict, proc_frame_idx: int, crop_scale: float = 0.40
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, (frame_idx / FPS_ASSUMED_BY_LIGHT_ASD) * 1000)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    proc = track["proc_track"]
    bs = proc["s"][proc_frame_idx]
    bsi = int(bs * (1 + 2 * crop_scale))
    padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
    my = proc["y"][proc_frame_idx] + bsi
    mx = proc["x"][proc_frame_idx] + bsi
    y0, y1 = int(my - bs), int(my + bs * (1 + 2 * crop_scale))
    x0, x1 = int(mx - bs * (1 + crop_scale)), int(mx + bs * (1 + crop_scale))
    return padded[y0:y1, x0:x1]


def _identify_character_track(
    video_path: str, tracks: list, face_app, target_embedding: np.ndarray, match_threshold: float
) -> tuple[int | None, float]:
    best_idx, best_score = None, -1.0
    for track_idx, track in enumerate(tracks):
        frame_ids = track["track"]["frame"]
        sample_positions = np.linspace(0, len(frame_ids) - 1, num=min(5, len(frame_ids)), dtype=int)
        similarities = []
        for pos in sample_positions:
            crop = _crop_face_at_frame(video_path, frame_ids[pos], track, pos)
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
        print(f"[LTXVLockCharacterVoice] track {track_idx}: identity similarity = {track_score:.3f}")
        if track_score > best_score:
            best_idx, best_score = track_idx, track_score
    if best_score < match_threshold:
        return None, best_score
    return best_idx, best_score


def _speaking_segments(
    track: dict, track_scores: list[float], speaking_threshold: float, min_segment_seconds: float
) -> list[tuple[float, float]]:
    frame_ids = track["track"]["frame"]
    is_speaking = [s > speaking_threshold for s in track_scores]
    segments = []
    seg_start = None
    for i, speaking in enumerate(is_speaking):
        if speaking and seg_start is None:
            seg_start = frame_ids[i]
        elif not speaking and seg_start is not None:
            segments.append((seg_start, frame_ids[i - 1] + 1))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, frame_ids[-1] + 1))
    fps = FPS_ASSUMED_BY_LIGHT_ASD
    return [(s / fps, e / fps) for s, e in segments if (e - s) / fps >= min_segment_seconds]


def _revoice_segments_in_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[tuple[float, float]],
    converter,
    target_se,
    work_dir: Path,
    tau: float,
    crossfade_seconds: float = 0.08,
) -> np.ndarray:
    """Re-voice only the given segments of a possibly multi-channel waveform ([C, T]), at its
    own original sample_rate throughout.

    OpenVoice's converter works on mono, so each segment is converted as mono (channels
    averaged) and the result is duplicated across all channels for JUST that segment's sample
    range. Audio outside the segments -- and each channel's original stereo image there -- is
    left completely untouched, rather than collapsing the whole track to mono.

    extract_se/convert internally resample whatever they're given to the model's own rate via
    librosa.load(sr=...), so segments are written at sample_rate as-is; only the returned
    converted audio (which convert() writes at the model's rate) needs resampling back.
    """
    import soundfile as sf
    import torchaudio

    audio = waveform.astype(np.float64).copy()  # [C, T]
    num_channels = audio.shape[0]
    crossfade_n = int(crossfade_seconds * sample_rate)
    model_sr = converter.hps.data.sampling_rate

    for i, (start_s, end_s) in enumerate(segments):
        start_i, end_i = int(start_s * sample_rate), int(end_s * sample_rate)
        segment_mono = audio[:, start_i:end_i].mean(axis=0) if num_channels > 1 else audio[0, start_i:end_i]
        seg_path = work_dir / f"segment_{i}.wav"
        sf.write(seg_path, segment_mono, sample_rate)

        src_se = converter.extract_se([str(seg_path)], se_save_path=None)
        converted_path = work_dir / f"segment_{i}.converted.wav"
        converter.convert(
            audio_src_path=str(seg_path), src_se=src_se, tgt_se=target_se, output_path=str(converted_path), tau=tau
        )
        converted, converted_sr = sf.read(converted_path)
        if converted_sr != sample_rate:
            converted = torchaudio.functional.resample(
                torch.from_numpy(converted.astype(np.float32)), converted_sr, sample_rate
            ).numpy()
        converted = converted.astype(np.float64)

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
        print(f"[LTXVLockCharacterVoice] re-voiced {start_s:.2f}s - {end_s:.2f}s (model rate {model_sr}Hz)")

    return audio


class LTXVLockCharacterVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "character_photo": ("IMAGE", {"tooltip": "One reference photo of the character's face."}),
                "character_voice": ("AUDIO", {"tooltip": "~5-10s reference clip of the character's target voice."}),
                "light_asd_repo": ("STRING", {"default": "custom_nodes/ltxv_voice_lock/third_party/Light-ASD"}),
                "light_asd_weight": (
                    "STRING",
                    {
                        "default": "weight/finetuning_TalkSet.model",
                        "tooltip": (
                            "Path relative to light_asd_repo (Columbia_test.py is run with that as its "
                            "working directory, so this must NOT be re-prefixed with light_asd_repo again)."
                        ),
                    },
                ),
                "converter_config": (
                    "STRING",
                    {"default": "custom_nodes/ltxv_voice_lock/checkpoints/converter/config.json"},
                ),
                "converter_ckpt": (
                    "STRING",
                    {"default": "custom_nodes/ltxv_voice_lock/checkpoints/converter/checkpoint.pth"},
                ),
                "match_threshold": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
                "speaking_threshold": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "min_segment_seconds": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.05}),
                "tau": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device": ("STRING", {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "execute"
    CATEGORY = "audio/ltxv"
    DESCRIPTION = (
        "Finds where an enrolled character is speaking (Light-ASD + InsightFace) and re-voices "
        "just those segments to match a reference voice sample (OpenVoice), leaving every other "
        "character's dialogue untouched. See this node package's README.md before first use."
    )

    def execute(  # noqa: PLR0913
        self,
        video,
        character_photo,
        character_voice,
        light_asd_repo,
        light_asd_weight,
        converter_config,
        converter_ckpt,
        match_threshold,
        speaking_threshold,
        min_segment_seconds,
        tau,
        device,
    ):
        from insightface.app import FaceAnalysis
        from openvoice.api import ToneColorConverter

        work_dir = Path(tempfile.mkdtemp(prefix="ltxv_voice_lock_"))
        components = video.get_components()
        if components.audio is None:
            print("[LTXVLockCharacterVoice] video has no audio track -- passing video through unchanged.")
            return (video,)

        video_path = str(work_dir / "input.mp4")
        video.save_to(video_path)

        photo_path = work_dir / "character.jpg"
        _save_photo(character_photo, photo_path)
        voice_path = work_dir / "character_voice.wav"
        _save_voice(character_voice, voice_path)

        print("[LTXVLockCharacterVoice] enrolling character face embedding...")
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        photo_img = cv2.imread(str(photo_path))
        faces = face_app.get(photo_img)
        if not faces:
            raise RuntimeError("No face detected in character_photo -- cannot enroll.")
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        target_embedding = largest.normed_embedding

        print("[LTXVLockCharacterVoice] running Light-ASD...")
        tracks, scores = _run_light_asd(video_path, light_asd_repo, light_asd_weight, work_dir)
        print(f"[LTXVLockCharacterVoice] found {len(tracks)} face track(s)")

        track_idx, match_score = _identify_character_track(
            video_path, tracks, face_app, target_embedding, match_threshold
        )
        if track_idx is None:
            print(
                f"[LTXVLockCharacterVoice] no track matched above threshold {match_threshold} "
                f"(best: {match_score:.3f}) -- passing video through unchanged."
            )
            return (video,)
        print(f"[LTXVLockCharacterVoice] matched track {track_idx} (similarity {match_score:.3f})")

        segments = _speaking_segments(tracks[track_idx], scores[track_idx], speaking_threshold, min_segment_seconds)
        if not segments:
            print("[LTXVLockCharacterVoice] no speaking segments found -- passing video through unchanged.")
            return (video,)
        print(f"[LTXVLockCharacterVoice] {len(segments)} segment(s) to re-voice")

        converter = ToneColorConverter(converter_config, device=device)
        converter.watermark_model = None  # see voice_lock/lock_character_voice.py for why this
        converter.load_ckpt(converter_ckpt)
        target_se = converter.extract_se([str(voice_path)], se_save_path=None)

        audio = components.audio
        sample_rate = int(audio["sample_rate"])  # audio is guaranteed non-None by the check above
        waveform = audio["waveform"][0].cpu().numpy()  # [C, T], original channel count preserved

        new_waveform = _revoice_segments_in_waveform(
            waveform, sample_rate, segments, converter, target_se, work_dir, tau
        )
        new_audio = {
            "waveform": torch.from_numpy(new_waveform.astype(np.float32)).unsqueeze(0),
            "sample_rate": sample_rate,
        }

        from comfy_api.latest import InputImpl, Types

        new_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=components.images, audio=new_audio, frame_rate=components.frame_rate)
        )
        print("[LTXVLockCharacterVoice] done.")
        return (new_video,)


NODE_CLASS_MAPPINGS = {"LTXVLockCharacterVoice": LTXVLockCharacterVoice}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXVLockCharacterVoice": "LTXV Lock Character Voice"}
