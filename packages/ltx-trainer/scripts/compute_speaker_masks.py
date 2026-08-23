#!/usr/bin/env python3
"""Precompute speech-token masks for LTX-2.5 speaker-adapter training.

The adapter should identify the speaker only while speech is present.  Music,
room tone, Foley and other non-speech audio should remain unconstrained so LTX
can learn/generate them natively.

This script transcribes the TARGET training media with faster-whisper word
timestamps and maps those word spans onto the already-computed LTX audio latent
time axis.  It writes one ``mask`` tensor of shape ``[T_audio_latent]`` per
sample, mirroring the relative paths used by ``audio_latents``.

A small time pad is applied around every word so consonant attacks, breaths and
short inter-word transitions are not accidentally excluded.  The output mask is
float32, which leaves room for future soft/feathered masks; this first version
writes 0/1 values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import track


console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
MEDIA_COLUMNS = ("media_path", "video")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError("JSON dataset must contain a top-level list")
        return rows
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _pick_media_column(rows: list[dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return explicit
    if not rows:
        raise ValueError("Dataset is empty")
    for column in MEDIA_COLUMNS:
        if column in rows[0]:
            return column
    raise KeyError(f"Could not find target media column; expected one of {MEDIA_COLUMNS}")


def _resolve(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def _relative(value: str, base: Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        return p
    try:
        return p.resolve().relative_to(base.resolve())
    except ValueError:
        return Path(p.name)


def _mask_from_words(
    words: list[Any],
    num_time_steps: int,
    duration: float,
    pad_seconds: float,
) -> torch.Tensor:
    if num_time_steps <= 0:
        raise ValueError(f"Invalid audio latent length: {num_time_steps}")
    if duration <= 0:
        raise ValueError(f"Invalid audio duration: {duration}")

    mask = torch.zeros(num_time_steps, dtype=torch.float32)
    seconds_per_token = duration / num_time_steps
    for word in words:
        start = max(0.0, float(word.start) - pad_seconds)
        end = min(duration, float(word.end) + pad_seconds)
        if end <= start:
            continue
        start_idx = max(0, min(num_time_steps - 1, int(start / seconds_per_token)))
        end_idx = max(start_idx + 1, min(num_time_steps, int(end / seconds_per_token) + 1))
        mask[start_idx:end_idx] = 1.0
    return mask


@app.command()
def main(
    dataset_file: Path = typer.Argument(..., exists=True, dir_okay=False),  # noqa: B008
    audio_latents_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--audio-latents-dir",
        help="Defaults to <dataset_dir>/.precomputed/audio_latents",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Defaults to <dataset_dir>/.precomputed/speaker_masks",
    ),
    media_column: str | None = typer.Option(None, "--media-column"),
    whisper_model: str = typer.Option("base.en", "--whisper-model"),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="CPU is safest beside LTX training; use cuda only during standalone preprocessing.",
    ),
    compute_type: str = typer.Option("int8", "--compute-type"),
    pad_seconds: float = typer.Option(0.08, "--pad-seconds", min=0.0, max=1.0),
    language: str = typer.Option("en", "--language"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Create one speech-region mask aligned to each target audio latent."""
    dataset_file = dataset_file.expanduser().resolve()
    base = dataset_file.parent
    rows = _load_rows(dataset_file)
    media_column = _pick_media_column(rows, media_column)
    audio_latents_dir = (
        audio_latents_dir.expanduser().resolve()
        if audio_latents_dir is not None
        else (base / ".precomputed" / "audio_latents").resolve()
    )
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (base / ".precomputed" / "speaker_masks").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_latents_dir.is_dir():
        raise FileNotFoundError(f"Audio latent directory does not exist: {audio_latents_dir}")

    console.print(
        f"Loading faster-whisper [cyan]{whisper_model}[/] on [cyan]{device}[/] "
        f"(compute_type={compute_type})..."
    )
    whisper = WhisperModel(whisper_model, device=device, compute_type=compute_type)

    written = skipped = failed = no_speech = 0
    for row in track(rows, description="Aligning speaker masks"):
        media_value = str(row[media_column])
        media_path = _resolve(media_value, base)
        rel = _relative(media_value, base).with_suffix(".pt")
        audio_latent_path = audio_latents_dir / rel
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        if not audio_latent_path.is_file():
            console.print(f"[red]Missing audio latent[/] {audio_latent_path}")
            failed += 1
            continue

        try:
            latent_meta = torch.load(audio_latent_path, map_location="cpu", weights_only=True)
            num_time_steps = int(latent_meta["num_time_steps"])
            duration_value = latent_meta["duration"]
            duration = float(duration_value.item() if isinstance(duration_value, torch.Tensor) else duration_value)

            segments, _ = whisper.transcribe(
                str(media_path),
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )
            words = [word for segment in segments for word in (segment.words or [])]
            mask = _mask_from_words(words, num_time_steps, duration, pad_seconds)
            if not words:
                no_speech += 1
                console.print(f"[yellow]No speech detected[/] {media_path.name}; writing an all-zero mask")

            torch.save(
                {
                    "mask": mask,
                    "num_time_steps": num_time_steps,
                    "duration": duration,
                    "word_count": len(words),
                    "whisper_model": whisper_model,
                    "pad_seconds": pad_seconds,
                },
                out_path,
            )
            written += 1
        except Exception as exc:
            console.print(f"[red]Failed[/] {media_path}: {exc}")
            failed += 1

    console.print(
        f"[green]Speaker masks complete[/]: written={written}, skipped={skipped}, "
        f"no_speech={no_speech}, failed={failed}, output={output_dir}"
    )


if __name__ == "__main__":
    app()
