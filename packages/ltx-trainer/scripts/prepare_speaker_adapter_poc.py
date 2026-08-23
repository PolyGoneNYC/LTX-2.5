#!/usr/bin/env python3
"""Prepare an audio-only LTX-2.5 speaker-adapter POC in one command.

This helper:
  1. Finds the trainable LTX-2.5 dev BF16 transformer, matching BF16 Gemma4
     text encoder, and audio VAE under a ComfyUI models directory.
  2. Runs official process_dataset.py for an audio-only T2A dataset.
  3. Computes the 512-D WavLM speaker embedding for every target clip.
  4. Writes a low-VRAM speaker-adapter YAML suitable for a 24 GB GPU smoke run.

It intentionally refuses Comfy-only int8/convrot weights because the LTX PyTorch
trainer cannot load those files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DEV_TRANSFORMER = "ltx-2.5-22b-dev-transformer-bf16.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"


def _find_one(root: Path, filename: str) -> Path:
    hits = list(root.rglob(filename))
    if not hits:
        raise FileNotFoundError(
            f"Missing required training file: {filename}\n"
            f"Looked under: {root}\n"
            "The *comfy-int8-convrot* files are not valid for LTX Trainer."
        )
    return hits[0].resolve()


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument(
        "--comfy-models",
        default="/workspace/runpod-slim/ComfyUI/models",
        help="ComfyUI models root",
    )
    parser.add_argument("--steps", type=int, default=50, help="Smoke-train steps written to config")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset_file).expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    dataset_root = dataset.parent
    precomputed = dataset_root / ".precomputed"
    comfy_models = Path(args.comfy_models).expanduser().resolve()

    transformer = _find_one(comfy_models, DEV_TRANSFORMER)
    text_encoder = _find_one(comfy_models, TEXT_ENCODER)
    audio_vae = _find_one(comfy_models, AUDIO_VAE)

    print("FOUND TRAINING ASSETS")
    print(f"  transformer: {transformer}")
    print(f"  text encoder: {text_encoder}")
    print(f"  audio VAE:    {audio_vae}")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    process_dataset = script_dir / "process_dataset.py"
    compute_speaker = script_dir / "compute_speaker_embeddings.py"

    preprocess_cmd = [
        sys.executable,
        str(process_dataset),
        str(dataset),
        "--model-path",
        str(transformer),
        "--text-encoder-path",
        str(text_encoder),
        "--audio-vae-path",
        str(audio_vae),
        "--batch-size",
        "1",
        "--load-text-encoder-in-8bit",
    ]
    if args.overwrite:
        preprocess_cmd.append("--overwrite")
    _run(preprocess_cmd)

    speaker_cmd = [
        sys.executable,
        str(compute_speaker),
        str(dataset),
        "--device",
        "cuda",
    ]
    if args.overwrite:
        speaker_cmd.append("--overwrite")
    _run(speaker_cmd)

    base_config = repo_root / "packages" / "ltx-trainer" / "configs" / "t2a_lora.yaml"
    with base_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["model_path"] = str(transformer)
    config["model"]["text_encoder_path"] = str(text_encoder)
    config["model"]["audio_vae_path"] = str(audio_vae)
    config["model"].pop("video_vae_path", None)
    config["model"]["training_mode"] = "lora"
    config["model"]["load_checkpoint"] = None

    config["training_strategy"] = {
        "name": "flexible",
        "audio": {"is_generated": True, "latents_dir": "audio_latents"},
    }

    config["optimization"]["learning_rate"] = float(args.learning_rate)
    config["optimization"]["steps"] = int(args.steps)
    config["optimization"]["batch_size"] = 1
    config["optimization"]["gradient_accumulation_steps"] = 1
    config["optimization"]["optimizer_type"] = "adamw8bit"
    config["optimization"]["enable_gradient_checkpointing"] = True

    # 32 GB is the documented INT8 low-VRAM target. For a 24 GB 4090 smoke run,
    # use INT4 for the frozen 22B base and train only the ~1M speaker projection.
    config["acceleration"]["mixed_precision_mode"] = "bf16"
    config["acceleration"]["quantization"] = "int4-quanto"
    config["acceleration"]["load_text_encoder_in_8bit"] = True

    config["data"]["preprocessed_data_root"] = str(precomputed)
    config["data"]["num_dataloader_workers"] = 0

    config["validation"]["interval"] = None
    config["validation"]["skip_initial_validation"] = True
    config["validation"]["generate_audio"] = False
    config["validation"]["generate_video"] = False

    config["checkpoints"]["interval"] = max(10, min(25, int(args.steps)))
    config["checkpoints"]["keep_last_n"] = 3
    config["wandb"]["enabled"] = False
    config["output_dir"] = str(dataset_root / "speaker_adapter_output")

    out_config = dataset_root / "speaker_adapter_24gb.yaml"
    with out_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\nPREPARATION COMPLETE")
    print(f"PRECOMPUTED: {precomputed}")
    print(f"CONFIG:      {out_config}")
    print("TRAIN WITH:")
    print(
        f"  uv run python packages/ltx-trainer/scripts/train_speaker_adapter.py {out_config} "
        "--speaker-mask-mode global"
    )


if __name__ == "__main__":
    main()
