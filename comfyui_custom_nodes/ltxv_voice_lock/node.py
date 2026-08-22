"""ComfyUI node: re-voice up to 4 enrolled characters' speaking segments in a generated video.

Runs entirely inside the ComfyUI graph -- no separate script, no output file to hand back in.
Wire it in right after wherever your workflow produces a VIDEO (e.g. the LTX-2.5 subgraph's
output) and before SaveVideo. Face detection and tracking (Light-ASD) runs once per video no
matter how many characters are enrolled -- only the per-character identity match and voice
conversion repeat.

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

import itertools
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


def _video_fps(video_path: str, fallback: float = 25.0) -> float:
    """Light-ASD's own frame extraction (ffmpeg -f image2, no -r/-vf fps) samples at the video's
    native fps, so track frame indices line up 1:1 with native-fps frame numbers. Re-seeking into
    the original video (for face crops) and converting frame ranges to seconds (for speaking
    segments) must use that same real fps, or both drift on any video that isn't exactly 25fps.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else fallback


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
    video_path: str, frame_idx: int, track: dict, proc_frame_idx: int, fps: float, crop_scale: float = 0.40
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, (frame_idx / fps) * 1000)
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
    video_path: str,
    tracks: list,
    face_app,
    target_embedding: np.ndarray,
    match_threshold: float,
    fps: float,
    exclude: set[int] | None = None,
) -> tuple[int | None, float]:
    exclude = exclude or set()
    best_idx, best_score = None, -1.0
    for track_idx, track in enumerate(tracks):
        if track_idx in exclude:
            continue
        frame_ids = track["track"]["frame"]
        sample_positions = np.linspace(0, len(frame_ids) - 1, num=min(5, len(frame_ids)), dtype=int)
        similarities = []
        empty_crops = no_faces = 0
        for pos in sample_positions:
            crop = _crop_face_at_frame(video_path, frame_ids[pos], track, pos, fps)
            if crop.size == 0:
                empty_crops += 1
                continue
            faces = face_app.get(crop)
            print(f"[LTXVLockCharacterVoice] track {track_idx} sample {pos}: crop shape = {crop.shape}")
            debug_dir = Path("/tmp/ltxv_voice_lock_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"track{track_idx}_sample{pos}.jpg"), crop)
            if not faces:
                no_faces += 1
                continue
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            similarities.append(float(np.dot(largest.normed_embedding, target_embedding)))
        if not similarities:
            print(
                f"[LTXVLockCharacterVoice] track {track_idx}: no usable face crop in "
                f"{len(sample_positions)} sample(s) (empty_crop={empty_crops}, no_face_detected={no_faces})"
            )
            continue
        track_score = float(np.median(similarities))
        print(f"[LTXVLockCharacterVoice] track {track_idx}: identity similarity = {track_score:.3f}")
        if track_score > best_score:
            best_idx, best_score = track_idx, track_score
    if best_score < match_threshold:
        return None, best_score
    return best_idx, best_score


def _speaking_segments(
    track: dict, track_scores: list[float], speaking_threshold: float, min_segment_seconds: float, fps: float
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
        # Segment timing comes from Light-ASD's frame-based tracking, which can run a few
        # samples past the actual audio buffer's length -- clamp so the write-back at the end
        # always matches the real slice length instead of silently clipping.
        end_i = min(end_i, audio.shape[1])
        if end_i <= start_i:
            continue
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


def _replace_segments_in_waveform(  # noqa: PLR0912
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[tuple[float, float]],
    separator,
    whisper_model,
    synthesize_voice,
    crossfade_seconds: float = 0.08,
) -> np.ndarray:
    """Fully replace (not blend) each given segment's voice, keeping any background sound.

    Per segment: separate vocals from background (Demucs), transcribe the isolated vocals
    (faster-whisper), re-synthesize that text from scratch in the target voice via the injected
    `synthesize_voice(text, target_duration_seconds) -> (mono_audio, sample_rate)` backend (either
    OpenVoice's base-speaker TTS + tone-color converter, or Fish Audio's direct zero-shot clone),
    time-fit the result to the segment's original duration, and rebuild the segment as
    new-voice + original background. Unlike the tone-color-only blend, nothing of the original
    voice recording survives into the output.
    """
    import torchaudio
    from librosa.effects import time_stretch

    audio = waveform.astype(np.float64).copy()  # [C, T]
    num_channels = audio.shape[0]
    crossfade_n = int(crossfade_seconds * sample_rate)
    demucs_sr = separator.samplerate

    for start_s, end_s in segments:
        start_i, end_i = int(start_s * sample_rate), int(end_s * sample_rate)
        # Segment timing comes from Light-ASD's frame-based tracking, which can run a few
        # samples past the actual audio buffer's length -- clamp so the write-back at the end
        # always matches the real slice length instead of silently clipping.
        end_i = min(end_i, audio.shape[1])
        if end_i <= start_i:
            continue
        target_len = end_i - start_i
        segment = audio[:, start_i:end_i]  # [C, T_seg]

        # 1. Separate vocals from background. Demucs is stereo-native; duplicate mono up front.
        seg_tensor = torch.from_numpy(segment).float()
        if num_channels == 1:
            seg_tensor = seg_tensor.repeat(2, 1)
        resampled_full, stems = separator.separate_tensor(seg_tensor, sr=sample_rate)
        vocals = stems["vocals"]  # [2, T_demucs_sr]
        background = resampled_full - vocals  # [2, T_demucs_sr]

        background_native = torchaudio.functional.resample(background, demucs_sr, sample_rate)
        if num_channels == 1:
            background_native = background_native.mean(dim=0, keepdim=True)
        background_native = background_native.cpu().numpy().astype(np.float64)
        if background_native.shape[1] > target_len:
            background_native = background_native[:, :target_len]
        elif background_native.shape[1] < target_len:
            pad = target_len - background_native.shape[1]
            background_native = np.pad(background_native, ((0, 0), (0, pad)))

        # 2. Transcribe the isolated vocals (faster-whisper wants 16kHz mono float32 when given
        #    a raw array directly -- it does NOT resample arrays itself, only file paths).
        vocals_mono = vocals.mean(dim=0)
        vocals_16k = torchaudio.functional.resample(vocals_mono, demucs_sr, 16000).cpu().numpy().astype(np.float32)
        whisper_segments, _info = whisper_model.transcribe(vocals_16k, language="en")
        text = "".join(s.text for s in whisper_segments).strip()
        if not text:
            print(f"[LTXVLockCharacterVoice] segment {start_s:.2f}s-{end_s:.2f}s: nothing transcribed, leaving as-is")
            continue

        # 3. Generate brand-new speech from that text in the target voice via the injected
        #    synthesis backend. The OpenVoice backend gets itself close to the segment's
        #    duration first via its own native `speed` parameter (the model actually pacing its
        #    delivery) rather than leaning on the phase-vocoder stretch in step 4 to do all the
        #    work -- large phase-vocoder stretches are what make replaced speech sound
        #    robotic/metallic, so minimizing how much stretching is needed afterward is the main
        #    lever for more natural-sounding output.
        target_duration = target_len / sample_rate
        converted, converted_sr = synthesize_voice(text, target_duration)
        converted = np.asarray(converted, dtype=np.float32)
        if converted_sr != sample_rate:
            converted = torchaudio.functional.resample(torch.from_numpy(converted), converted_sr, sample_rate).numpy()

        # 4. Time-fit the new speech to the segment's original duration (pitch-preserving). The
        #    speed adjustment in step 3 should have already gotten close, so only phase-vocoder
        #    stretch the small residual mismatch -- skip it entirely when close enough, since
        #    stretching (even a little) is the main source of robotic-sounding artifacts.
        if len(converted) > 0:
            rate = len(converted) / target_len
            if abs(rate - 1.0) > 0.03:
                rate = min(max(rate, 0.4), 2.5)  # avoid pathological stretching on extreme mismatches
                converted = time_stretch(converted, rate=rate)
        converted = converted.astype(np.float64)
        if len(converted) > target_len:
            converted = converted[:target_len]
        elif len(converted) < target_len:
            converted = np.pad(converted, (0, target_len - len(converted)))

        # 5. Rebuild the segment: new voice on top of the original background, nothing of the
        #    original voice recording left in it. Crossfade the edges against the ORIGINAL
        #    segment audio so the cut in/out doesn't click.
        new_segment = np.tile(converted, (num_channels, 1)) + background_native
        n = min(crossfade_n, target_len // 2)
        if n > 0:
            fade_in = np.linspace(0, 1, n)
            fade_out = 1 - fade_in
            new_segment[:, :n] = new_segment[:, :n] * fade_in + segment[:, :n] * fade_out
            new_segment[:, -n:] = new_segment[:, -n:] * fade_out + segment[:, -n:] * fade_in

        audio[:, start_i:end_i] = new_segment
        print(f'[LTXVLockCharacterVoice] replaced {start_s:.2f}s-{end_s:.2f}s: "{text}"')

    return audio


def _make_openvoice_synthesizer(base_tts, base_source_se, converter, target_se, tau, work_dir: Path):
    """Build a synthesize_voice(text, target_duration_s) backend using OpenVoice's own
    base-speaker TTS (to generate the words) followed by its tone-color converter (to shift
    the timbre toward the target voice). Returns (mono_audio, sample_rate).
    """
    import soundfile as sf

    counter = itertools.count()

    def synth(text: str, target_duration_s: float) -> tuple[np.ndarray, int]:
        i = next(counter)
        tts_path = work_dir / f"openvoice_{i}_tts.wav"
        base_tts.tts(text, str(tts_path), speaker="default", language="English", speed=1.0)
        probe, probe_sr = sf.read(str(tts_path))
        probe_duration = len(probe) / probe_sr if probe_sr else 0.0
        if probe_duration > 0 and target_duration_s > 0:
            speed = probe_duration / target_duration_s
            speed = min(max(speed, 0.7), 1.3)
            if abs(speed - 1.0) > 0.03:
                base_tts.tts(text, str(tts_path), speaker="default", language="English", speed=speed)
        converted_path = work_dir / f"openvoice_{i}_converted.wav"
        converter.convert(
            audio_src_path=str(tts_path),
            src_se=base_source_se,
            tgt_se=target_se,
            output_path=str(converted_path),
            tau=tau,
        )
        converted, converted_sr = sf.read(str(converted_path))
        return converted.astype(np.float32), converted_sr

    return synth


# fish-speech sizes its KV cache for the model's own max_seq_len (commonly 32768, meant for
# long-form use), which allocates several GB of VRAM regardless of how short the actual
# request is. Every segment this node sends through is a few seconds at most, so clamp it
# down -- still far more headroom than any realistic reference-clip + generated-segment length
# needs, while cutting that cache allocation roughly 8x.
_FISH_MAX_SEQ_LEN = 4096

# launch_thread_safe_queue() (below) spawns a daemon thread that loads the Llama model onto
# the GPU and loops forever processing requests -- it has no automatic shutdown, so calling
# _load_fish_engine() fresh on every node execution leaks a whole new model + thread each run
# (each is only reclaimed if the process exits). Cache by checkpoint identity so repeat runs
# with the same checkpoints reuse the already-loaded engine instead of leaking another one.
_FISH_ENGINE_CACHE: dict[tuple[str, str, str, str], object] = {}


def _load_fish_engine(llama_checkpoint: str, decoder_checkpoint: str, decoder_config_name: str, device: str):
    """Load (or reuse a cached) Fish Audio TTS inference engine (zero-shot voice cloning, single-pass)."""
    cache_key = (llama_checkpoint, decoder_checkpoint, decoder_config_name, device)
    cached = _FISH_ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_fish_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.text2semantic.llama import DualARTransformer

    if not getattr(DualARTransformer.setup_caches, "_ltxv_clamped", False):
        original_setup_caches = DualARTransformer.setup_caches

        def _clamped_setup_caches(self, max_batch_size, max_seq_len, dtype=torch.bfloat16):
            return original_setup_caches(self, max_batch_size, min(max_seq_len, _FISH_MAX_SEQ_LEN), dtype)

        _clamped_setup_caches._ltxv_clamped = True
        DualARTransformer.setup_caches = _clamped_setup_caches

    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=llama_checkpoint, device=device, precision=precision, compile=False
    )
    decoder_model = load_fish_decoder_model(
        config_name=decoder_config_name, checkpoint_path=decoder_checkpoint, device=device
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue, decoder_model=decoder_model, precision=precision, compile=False
    )
    _FISH_ENGINE_CACHE[cache_key] = engine
    return engine


def _make_fishaudio_synthesizer(engine, ref_audio_bytes: bytes, ref_text: str):
    """Build a synthesize_voice(text, target_duration_s) backend using Fish Audio's direct
    zero-shot voice cloning (reference audio + its transcript, in one pass -- no separate
    tone-color conversion step). Returns (mono_audio, sample_rate).
    """
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    def synth(text: str, target_duration_s: float) -> tuple[np.ndarray, int]:  # noqa: ARG001
        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text)],
        )
        for result in engine.inference(req):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final":
                sr, audio_np = result.audio
                return np.asarray(audio_np, dtype=np.float32), sr
        raise RuntimeError("Fish Audio produced no audio output")

    return synth


MAX_CHARACTERS = 4


class LTXVLockCharacterVoice:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video": ("VIDEO",),
            "character_photo": ("IMAGE", {"tooltip": "Character 1's reference face photo."}),
            "character_voice": ("AUDIO", {"tooltip": "Character 1's ~5-10s reference voice clip."}),
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
            "mode": (
                ["replace_fishaudio", "replace", "blend"],
                {
                    "default": "replace_fishaudio",
                    "tooltip": (
                        "replace_fishaudio: transcribes each segment and re-synthesizes it from scratch "
                        "using Fish Audio's zero-shot voice cloning (reference clip + text, one pass), "
                        "keeping background sound but replacing 100% of the original voice recording. "
                        "The most natural-sounding option. Needs the fish_*/whisper_model inputs below. "
                        "replace: same idea, but using OpenVoice's base-speaker TTS + tone-color "
                        "converter instead -- kept as a fallback; tends to sound more robotic. Needs the "
                        "base_speaker_*/whisper_model inputs below. "
                        "blend: the original tone-color-only conversion -- keeps the original "
                        "recording's delivery/rhythm and only shifts its timbre toward the target "
                        "voice, so some of the original voice's character always comes through."
                    ),
                },
            ),
            "whisper_model": (
                "STRING",
                {
                    "default": "base.en",
                    "tooltip": "faster-whisper model size for transcribing segments in either replace mode.",
                },
            ),
            "base_speaker_config": (
                "STRING",
                {"default": "custom_nodes/ltxv_voice_lock/checkpoints/base_speakers/EN/config.json"},
            ),
            "base_speaker_ckpt": (
                "STRING",
                {"default": "custom_nodes/ltxv_voice_lock/checkpoints/base_speakers/EN/checkpoint.pth"},
            ),
            "base_speaker_se": (
                "STRING",
                {"default": "custom_nodes/ltxv_voice_lock/checkpoints/base_speakers/EN/en_default_se.pth"},
            ),
            "fish_llama_checkpoint": (
                "STRING",
                {
                    "default": "custom_nodes/ltxv_voice_lock/checkpoints/fish_s2pro",
                    "tooltip": "Directory containing Fish Audio's semantic-token model weights.",
                },
            ),
            "fish_decoder_checkpoint": (
                "STRING",
                {"default": "custom_nodes/ltxv_voice_lock/checkpoints/fish_s2pro/codec.pth"},
            ),
            "fish_decoder_config_name": ("STRING", {"default": "modded_dac_vq"}),
            "match_threshold": (
                "FLOAT",
                {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Applies to every character."},
            ),
            "speaking_threshold": (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "Applies to every character."},
            ),
            "min_segment_seconds": (
                "FLOAT",
                {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Applies to every character."},
            ),
            "tau": (
                "FLOAT",
                {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Applies to every character."},
            ),
            "device": ("STRING", {"default": "cuda:0"}),
        }
        optional = {}
        for i in range(2, MAX_CHARACTERS + 1):
            optional[f"character_photo_{i}"] = ("IMAGE", {"tooltip": f"Character {i}'s reference face photo."})
            optional[f"character_voice_{i}"] = ("AUDIO", {"tooltip": f"Character {i}'s reference voice clip."})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "execute"
    CATEGORY = "audio/ltxv"
    DESCRIPTION = (
        "Finds where each enrolled character (up to 4) is speaking (Light-ASD + InsightFace), "
        "leaving every unenrolled character's dialogue untouched. 'replace_fishaudio' and "
        "'replace' modes both transcribe each segment and re-synthesize it from scratch in the "
        "reference voice, keeping background sound but replacing the original voice entirely -- "
        "they only differ in which TTS/cloning engine generates the new speech (Fish Audio's "
        "direct zero-shot clone vs. OpenVoice's base-speaker TTS + tone-color converter). 'blend' "
        "mode is the original tone-color-only conversion, which keeps some of the original "
        "recording's delivery. Face detection runs once for the whole video regardless of how "
        "many characters are enrolled. See this node package's README.md before first use."
    )

    def execute(  # noqa: PLR0912, PLR0913
        self,
        video,
        character_photo,
        character_voice,
        light_asd_repo,
        light_asd_weight,
        converter_config,
        converter_ckpt,
        mode,
        whisper_model,
        base_speaker_config,
        base_speaker_ckpt,
        base_speaker_se,
        fish_llama_checkpoint,
        fish_decoder_checkpoint,
        fish_decoder_config_name,
        match_threshold,
        speaking_threshold,
        min_segment_seconds,
        tau,
        device,
        character_photo_2=None,
        character_voice_2=None,
        character_photo_3=None,
        character_voice_3=None,
        character_photo_4=None,
        character_voice_4=None,
    ):
        from insightface.app import FaceAnalysis
        from openvoice.api import ToneColorConverter

        characters = [(character_photo, character_voice)]
        for photo, voice in (
            (character_photo_2, character_voice_2),
            (character_photo_3, character_voice_3),
            (character_photo_4, character_voice_4),
        ):
            if photo is not None and voice is not None:
                characters.append((photo, voice))

        work_dir = Path(tempfile.mkdtemp(prefix="ltxv_voice_lock_"))
        components = video.get_components()
        if components.audio is None:
            print("[LTXVLockCharacterVoice] video has no audio track -- passing video through unchanged.")
            return (video,)

        video_path = str(work_dir / "input.mp4")
        video.save_to(video_path)
        video_fps = _video_fps(video_path)

        # This node's models (InsightFace, Demucs, Fish Audio, etc.) are loaded with plain
        # PyTorch, not through ComfyUI's own ModelPatcher/model-manager -- so ComfyUI has no way
        # to know they need VRAM and won't auto-evict the (still-resident) video-generation
        # models to make room. Free that memory ourselves before loading anything of our own;
        # the video/audio content is already fully extracted into `components` above, so nothing
        # downstream still needs those models loaded.
        if torch.cuda.is_available():
            before_gb = torch.cuda.memory_allocated() / 1e9
        try:
            from comfy import model_management

            model_management.unload_all_models()
            model_management.soft_empty_cache()
        except ImportError:
            pass
        if torch.cuda.is_available():
            after_gb = torch.cuda.memory_allocated() / 1e9
            print(
                f"[LTXVLockCharacterVoice] GPU memory allocated before/after unload_all_models: "
                f"{before_gb:.2f}GB -> {after_gb:.2f}GB"
            )

        print(f"[LTXVLockCharacterVoice] enrolling {len(characters)} character face embedding(s)...")
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        enrolled = []  # list of (character_number, target_embedding, voice_path)
        for i, (photo, voice) in enumerate(characters, start=1):
            photo_path = work_dir / f"character_{i}.jpg"
            _save_photo(photo, photo_path)
            voice_path = work_dir / f"character_{i}_voice.wav"
            _save_voice(voice, voice_path)

            photo_img = cv2.imread(str(photo_path))
            faces = face_app.get(photo_img)
            if not faces:
                print(f"[LTXVLockCharacterVoice] character {i}: no face detected in photo -- skipping.")
                continue
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            enrolled.append((i, largest.normed_embedding, voice_path))

        if not enrolled:
            print("[LTXVLockCharacterVoice] no character faces enrolled -- passing video through unchanged.")
            return (video,)

        print("[LTXVLockCharacterVoice] running Light-ASD...")
        tracks, scores = _run_light_asd(video_path, light_asd_repo, light_asd_weight, work_dir)
        print(f"[LTXVLockCharacterVoice] found {len(tracks)} face track(s)")

        converter = None
        if mode != "replace_fishaudio":
            converter = ToneColorConverter(converter_config, device=device)
            converter.watermark_model = None  # see voice_lock/lock_character_voice.py for why this
            converter.load_ckpt(converter_ckpt)

        separator = whisper = base_tts = base_source_se = fish_engine = None
        if mode in ("replace", "replace_fishaudio"):
            from demucs.api import Separator
            from faster_whisper import WhisperModel

            print(f"[LTXVLockCharacterVoice] loading vocal-separation/transcription models for {mode} mode...")
            separator = Separator(model="htdemucs", device=device)
            # faster-whisper's CUDA backend (ctranslate2) needs CUDA 12.x runtime libs
            # (libcublas.so.12 etc.) which aren't guaranteed present alongside newer CUDA
            # stacks (e.g. cu13 torch builds) -- CPU is plenty fast for these small models.
            whisper = WhisperModel(whisper_model, device="cpu")

        if mode == "replace":
            from openvoice.api import BaseSpeakerTTS

            base_tts = BaseSpeakerTTS(base_speaker_config, device=device)
            base_tts.load_ckpt(base_speaker_ckpt)
            base_source_se = torch.load(base_speaker_se, map_location=device)
        elif mode == "replace_fishaudio":
            print("[LTXVLockCharacterVoice] loading Fish Audio voice-cloning engine...")
            fish_engine = _load_fish_engine(
                fish_llama_checkpoint, fish_decoder_checkpoint, fish_decoder_config_name, device
            )

        audio = components.audio
        sample_rate = int(audio["sample_rate"])  # audio is guaranteed non-None by the check above
        waveform = audio["waveform"][0].cpu().numpy()  # [C, T], original channel count preserved

        claimed_tracks: set[int] = set()
        any_revoiced = False
        for i, target_embedding, voice_path in enrolled:
            track_idx, match_score = _identify_character_track(
                video_path, tracks, face_app, target_embedding, match_threshold, video_fps, exclude=claimed_tracks
            )
            if track_idx is None:
                print(
                    f"[LTXVLockCharacterVoice] character {i}: no track matched above threshold "
                    f"{match_threshold} (best: {match_score:.3f}) -- skipping."
                )
                continue
            claimed_tracks.add(track_idx)
            print(f"[LTXVLockCharacterVoice] character {i}: matched track {track_idx} (similarity {match_score:.3f})")

            segments = _speaking_segments(
                tracks[track_idx], scores[track_idx], speaking_threshold, min_segment_seconds, video_fps
            )
            if not segments:
                print(f"[LTXVLockCharacterVoice] character {i}: no speaking segments found -- skipping.")
                continue
            print(f"[LTXVLockCharacterVoice] character {i}: {len(segments)} segment(s) to {mode}")

            if mode == "replace_fishaudio":
                ref_segments, _ = whisper.transcribe(str(voice_path), language="en")
                ref_text = "".join(s.text for s in ref_segments).strip()
                synth = _make_fishaudio_synthesizer(fish_engine, voice_path.read_bytes(), ref_text)
                waveform = _replace_segments_in_waveform(waveform, sample_rate, segments, separator, whisper, synth)
            elif mode == "replace":
                target_se = converter.extract_se([str(voice_path)], se_save_path=None)
                synth = _make_openvoice_synthesizer(base_tts, base_source_se, converter, target_se, tau, work_dir)
                waveform = _replace_segments_in_waveform(waveform, sample_rate, segments, separator, whisper, synth)
            else:
                target_se = converter.extract_se([str(voice_path)], se_save_path=None)
                waveform = _revoice_segments_in_waveform(
                    waveform, sample_rate, segments, converter, target_se, work_dir, tau
                )
            any_revoiced = True

        if not any_revoiced:
            print("[LTXVLockCharacterVoice] nothing re-voiced -- passing video through unchanged.")
            return (video,)

        new_audio = {
            "waveform": torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0),
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
