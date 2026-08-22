#!/usr/bin/env python3
"""Decide one question: can LTX-2.5 generate speech that lip-syncs to EXISTING footage?

Background
----------
Our ComfyUI voice-lock node replaces voices *after* generation: separate vocals (Demucs),
transcribe (whisper), re-synthesize (TTS), time-fit (phase vocoder), remix. That chain
inverts two hard problems and leaks on both -- residual original voice, and flat/dry speech
with none of the scene's acoustics.

LTX-2.5 can instead generate the audio *inside* the diffusion loop with the video pinned
clean. `RetakePipeline(regenerate_video=False, regenerate_audio=True)` does exactly that:
the source video latent is frozen at sigma=0 (an exact fixed point -- see
`ltx_core/components/noisers.py` lerp on denoise_mask, and `helpers.py` post_process_latent),
while the audio latent denoises from pure noise, attending to those clean video tokens on
every step through the unmasked `video_to_audio_attn` path in every transformer block
(`ltx_core/model/transformer/transformer.py`), on a shared seconds-based RoPE time axis.

If that cross-attention is strong enough, the model times its words to the mouth it can see
and we get native lip sync, zero voice bleed, and scene-appropriate acoustics for free.

Nothing in the repo proves it is strong enough. Every shipped pipeline runs the OTHER
direction (generate audio, then regenerate video to match it). This script answers the
question empirically.

NOTE: the `retake` CLI cannot do this -- its `main()` never passes `regenerate_video`, so it
always regenerates both streams. This has to go through the Python API.

Usage
-----
  # 0. one-time: make a compliant test clip from any source video
  python native_v2a_test.py prep --input raw.mp4 --output clip.mp4

  # 1. the experiment
  python native_v2a_test.py run \
      --video clip.mp4 \
      --prompt "A man says: What the hell are you looking at?" \
      --models-root /workspace/runpod-slim/ComfyUI/models

  # 2. if sync is weak, turn on cross-modal guidance (the documented lipsync dial)
  python native_v2a_test.py run --video clip.mp4 --prompt "..." \
      --models-root ... --guidance 2.0

Then watch out.mp4 and answer: does it speak to the mouth, or just talk over the footage?
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# LTX requires frame count 8k+1 and width/height multiples of 32 (see retake.py main()).
TEMPORAL_GRID = 8
SPATIAL_GRID = 32
# audio_positional_embedding_max_pos defaults to [20]; beyond ~20s we are outside the
# trained positional range for the cross-modal RoPE.
MAX_TRAINED_SECONDS = 20.0


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def probe(path: str) -> dict:
    """ffprobe the video: frames, dims, fps, whether it has an audio stream."""
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-count_frames",
            str(path),
        ]
    )
    streams = json.loads(out)["streams"]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    num, den = (video.get("r_frame_rate") or "0/1").split("/")
    return {
        "frames": int(video.get("nb_read_frames") or 0),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "has_audio": audio is not None,
    }


def cmd_prep(args: argparse.Namespace) -> int:
    """Trim/crop any source video into something RetakePipeline will accept."""
    meta = probe(args.input)
    print(f"source: {meta['frames']} frames, {meta['width']}x{meta['height']}, {meta['fps']:.3f} fps")

    target_frames = args.frames
    if (target_frames - 1) % TEMPORAL_GRID != 0:
        raise SystemExit(f"--frames must satisfy 8k+1 (97, 193, 241, ...); got {target_frames}")
    if target_frames > meta["frames"]:
        raise SystemExit(f"source has only {meta['frames']} frames, need >= {target_frames}")

    # Crop (not scale) down to multiples of 32 so we don't resample the pixels.
    w = (meta["width"] // SPATIAL_GRID) * SPATIAL_GRID
    h = (meta["height"] // SPATIAL_GRID) * SPATIAL_GRID
    if w == 0 or h == 0:
        raise SystemExit(f"video too small to crop to a multiple of {SPATIAL_GRID}")

    duration = target_frames / meta["fps"] if meta["fps"] else 0
    if duration > MAX_TRAINED_SECONDS:
        print(f"WARNING: {duration:.1f}s exceeds the ~{MAX_TRAINED_SECONDS:.0f}s trained range")

    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.input),
            "-vf",
            f"crop={w}:{h}:0:0",
            "-frames:v",
            str(target_frames),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(args.output),
        ]
    )
    got = probe(args.output)
    print(
        f"wrote {args.output}: {got['frames']} frames, {got['width']}x{got['height']}, "
        f"{got['fps']:.3f} fps, audio={got['has_audio']}"
    )
    if not got["has_audio"]:
        print(
            "NOTE: no audio stream. Without one the pipeline regenerates the whole audio\n"
            "      track rather than a masked time region -- fine for this test, but the\n"
            "      TemporalRegionMask path stays unexercised."
        )
    return 0


def preflight(video: str) -> dict:
    meta = probe(video)
    problems = []
    if (meta["frames"] - 1) % TEMPORAL_GRID != 0:
        snapped = ((meta["frames"] - 1) // TEMPORAL_GRID) * TEMPORAL_GRID + 1
        problems.append(
            f"frame count must be 8k+1; got {meta['frames']} (nearest valid: {snapped}). "
            f"Fix with: {sys.argv[0]} prep --input {video} --output clip.mp4 --frames {snapped}"
        )
    if meta["width"] % SPATIAL_GRID or meta["height"] % SPATIAL_GRID:
        problems.append(f"width/height must be multiples of {SPATIAL_GRID}; got {meta['width']}x{meta['height']}")
    if problems:
        raise SystemExit("Source video is not compatible:\n  - " + "\n  - ".join(problems))
    return meta


def resolve_model_paths(root: Path | None, explicit: dict[str, str | None]):
    """Build ModelPaths, auto-discovering a ComfyUI-style split layout when possible."""
    from ltx_pipelines.utils.model_paths import ModelPaths

    def find(subdir: str, *patterns: str) -> str | None:
        if root is None:
            return None
        d = root / subdir
        if not d.is_dir():
            return None
        for pat in patterns:
            hits = sorted(d.glob(pat))
            if hits:
                return str(hits[0])
        return None

    transformer = explicit["transformer"] or find(
        "diffusion_models", "*2.5*distilled*transformer*.safetensors", "*distilled*transformer*.safetensors"
    )
    text_encoder = explicit["text_encoder"] or find("text_encoders", "*gemma*", "*.safetensors")
    video_vae = explicit["video_vae"] or find("vae", "*video-vae*conv*.safetensors", "*video*vae*.safetensors")
    audio_vae = explicit["audio_vae"] or find("vae", "*audio*vae*.safetensors")

    missing = [
        n
        for n, v in [
            ("transformer", transformer),
            ("text_encoder", text_encoder),
            ("video_vae", video_vae),
            ("audio_vae", audio_vae),
        ]
        if not v
    ]
    if missing:
        raise SystemExit(
            "Could not resolve model paths: " + ", ".join(missing) + "\n"
            "Pass --models-root pointing at your ComfyUI models dir, or set each explicitly\n"
            "with --transformer/--text-encoder/--video-vae/--audio-vae.\n"
            "NOTE: RetakePipeline needs the DISTILLED transformer\n"
            "      (ltx-2.5-22b-distilled-transformer-bf16.safetensors)."
        )

    for name, p in [
        ("transformer", transformer),
        ("text_encoder", text_encoder),
        ("video_vae", video_vae),
        ("audio_vae", audio_vae),
    ]:
        print(f"  {name:13s} {p}")

    return ModelPaths.from_split(
        transformer_path=transformer,
        text_encoder_path=text_encoder,
        video_vae_path=video_vae,
        audio_vae_path=audio_vae,
    )


def save_wav(audio_obj, path: Path) -> int:
    """AudioDecoder returns ltx_core.types.Audio(waveform, sampling_rate)."""
    import numpy as np

    waveform = getattr(audio_obj, "waveform", audio_obj)
    sr = int(getattr(audio_obj, "sampling_rate", 48000))

    arr = waveform.detach().float().cpu().numpy()
    while arr.ndim > 2:  # drop leading batch dims
        arr = arr[0]
    if arr.ndim == 2:  # [C, T] -> [T, C] for soundfile
        arr = arr.T
    arr = np.clip(arr, -1.0, 1.0)

    import soundfile as sf

    sf.write(str(path), arr, sr)
    return sr


def cmd_run(args: argparse.Namespace) -> int:
    meta = preflight(args.video)
    duration = meta["frames"] / meta["fps"]
    print(
        f"source: {meta['frames']} frames, {meta['width']}x{meta['height']}, "
        f"{meta['fps']:.3f} fps, {duration:.2f}s, audio={meta['has_audio']}"
    )
    if duration > MAX_TRAINED_SECONDS:
        print(f"WARNING: {duration:.1f}s exceeds the ~{MAX_TRAINED_SECONDS:.0f}s trained positional range")

    end_time = args.end_time if args.end_time is not None else duration
    print(f"regenerating audio over [{args.start_time:.2f}s, {end_time:.2f}s], video FROZEN")

    import torch

    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.utils.types import OffloadMode

    print("resolving model paths...")
    model_paths = resolve_model_paths(
        Path(args.models_root) if args.models_root else None,
        {
            "transformer": args.transformer,
            "text_encoder": args.text_encoder,
            "video_vae": args.video_vae,
            "audio_vae": args.audio_vae,
        },
    )

    offload = {
        "none": OffloadMode.NONE,
        "cpu": OffloadMode.CPU,
        "disk": OffloadMode.DISK,
    }[args.offload]

    # guidance != off needs the full (non-distilled) denoiser: RetakePipeline only builds a
    # MultiModalGuider when distilled=False. modality_scale is the documented lipsync dial
    # (--v2a-guidance-scale: "Higher values may increase lipsync quality").
    use_guidance = args.guidance is not None
    if use_guidance:
        print(
            f"guidance ON (modality_scale={args.guidance}) -> non-distilled denoiser, "
            f"{args.steps} steps. Expect ~2x time and VRAM."
        )

    print("building pipeline (first run loads ~20GB of weights, be patient)...")
    t0 = time.time()
    pipeline = RetakePipeline(
        model_paths=model_paths,
        loras=(),
        distilled=not use_guidance,
        offload_mode=offload,
    )
    print(f"pipeline ready in {time.time() - t0:.1f}s")

    call_kwargs = {}
    if use_guidance:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.utils.constants import detect_params

        params = detect_params(model_paths.transformer())
        base_audio = params.audio_guider_params
        call_kwargs["video_guider_params"] = params.video_guider_params
        call_kwargs["audio_guider_params"] = MultiModalGuiderParams(
            cfg_scale=getattr(base_audio, "cfg_scale", 1.0),
            modality_scale=args.guidance,
        )
        call_kwargs["num_inference_steps"] = args.steps

    print("running diffusion...")
    t0 = time.time()
    _video_iter, audio, _tiling = pipeline(
        video_path=str(args.video),
        prompt=args.prompt,
        start_time=args.start_time,
        end_time=end_time,
        seed=args.seed,
        regenerate_video=False,  # <-- the whole point: freeze the video stream
        regenerate_audio=True,
        max_batch_size=1,
        **call_kwargs,
    )
    elapsed = time.time() - t0
    print(f"diffusion done in {elapsed:.1f}s")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # The returned video is only decode(encode(source)) -- a VAE round-trip of footage we
    # already have, strictly softer than the original. Discard it and mux the new audio onto
    # the untouched source instead: pixel-perfect video, and we skip the video decode.
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = Path(args.output).with_suffix(".wav")
    sr = save_wav(audio, wav_path)
    print(f"wrote {wav_path} ({sr} Hz)")

    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.video),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(args.output),
        ]
    )
    print(f"wrote {args.output}")
    print(
        "\n--- THE QUESTION ---\n"
        "Watch it. Does the generated speech land on the mouth movements, or does it just\n"
        "talk over the footage at its own pace?\n"
        "  good sync  -> the native path works; next is voice identity (see step 5 of the plan)\n"
        "  weak sync  -> retry with --guidance 2.0 (then 2.5 / 4.0) before concluding\n"
        "  no sync    -> V2A can't drive a frozen video; the post-processing node stays\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test whether LTX-2.5 can generate lip-synced speech over frozen video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prep", help="trim/crop a source video into a compliant test clip")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="clip.mp4")
    p.add_argument("--frames", type=int, default=97, help="8k+1: 97 (~4s@24fps), 193, 241...")
    p.set_defaults(func=cmd_prep)

    r = sub.add_parser("run", help="run the frozen-video / regenerate-audio experiment")
    r.add_argument("--video", required=True, help="source clip (8k+1 frames, dims %%32)")
    r.add_argument("--prompt", required=True, help="describe the speech AND the voice")
    r.add_argument("--output", default="out.mp4")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--start-time", type=float, default=0.0)
    r.add_argument("--end-time", type=float, default=None, help="default: end of clip")
    r.add_argument("--models-root", default=None, help="e.g. /workspace/runpod-slim/ComfyUI/models")
    r.add_argument("--transformer", default=None, help="must be the DISTILLED transformer")
    r.add_argument("--text-encoder", default=None)
    r.add_argument("--video-vae", default=None)
    r.add_argument("--audio-vae", default=None)
    r.add_argument(
        "--offload",
        choices=["none", "cpu", "disk"],
        default="cpu",
        help="cpu: ~5GB VRAM + ~36GB RAM (default). none: ~28GB VRAM, faster.",
    )
    r.add_argument(
        "--guidance", type=float, default=None, help="cross-modal guidance (modality_scale), the lipsync dial. Try 2.0."
    )
    r.add_argument("--steps", type=int, default=40, help="only used with --guidance")
    r.set_defaults(func=cmd_run)

    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
