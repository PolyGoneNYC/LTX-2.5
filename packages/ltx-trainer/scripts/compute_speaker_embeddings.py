#!/usr/bin/env python3
"""Precompute fixed speaker embeddings for LTX-2.5 voice-adapter training.

Each dataset row points to:
  * the normal target media used by LTX training (``media_path`` or ``video``), and
  * a separate clean reference recording in ``speaker_reference``.

The reference recording SHOULD be a different utterance from the target clip.  That
forces the adapter to learn speaker identity rather than memorizing the target words,
prosody, or room acoustics.

The output directory mirrors the normal latent directory layout, so
``PrecomputedDataset`` can pair files by relative path:

    .precomputed/latents/scene_001.pt
    .precomputed/speaker_embeddings/scene_001.pt

Default encoder: microsoft/wavlm-base-plus-sv (speaker-verification X-vectors).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
import typer
from rich.console import Console
from rich.progress import track
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

MEDIA_COLUMNS = ("media_path", "video")
DEFAULT_SPEAKER_COLUMN = "speaker_reference"
DEFAULT_MODEL = "microsoft/wavlm-base-plus-sv"
TARGET_SAMPLE_RATE = 16000


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must contain a top-level list of rows")
        return data
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported dataset format: {path.suffix}; expected .json, .jsonl or .csv")


def _media_column(rows: list[dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return explicit
    if not rows:
        raise ValueError("Dataset is empty")
    for column in MEDIA_COLUMNS:
        if column in rows[0]:
            return column
    raise KeyError(f"Could not find a target-media column. Expected one of {MEDIA_COLUMNS}")


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _output_relative(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return path
    try:
        return path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        # Absolute media outside the dataset tree still needs a deterministic safe name.
        return Path(path.name)


def _load_reference_audio(path: Path, max_seconds: float) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio: {path}")
    waveform = waveform.float().mean(dim=0)  # mono [T]
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    if max_seconds > 0:
        waveform = waveform[: int(max_seconds * TARGET_SAMPLE_RATE)]
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak
    return waveform


@torch.inference_mode()
def _embed(
    waveform: torch.Tensor,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: WavLMForXVector,
    device: torch.device,
) -> torch.Tensor:
    inputs = feature_extractor(
        waveform.cpu().numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    embedding = model(input_values=input_values, attention_mask=attention_mask).embeddings
    embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    return embedding[0].cpu().contiguous()


@app.command()
def main(
    dataset_file: Path = typer.Argument(..., exists=True, dir_okay=False),  # noqa: B008
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Defaults to <dataset_dir>/.precomputed/speaker_embeddings",
    ),
    speaker_column: str = typer.Option(DEFAULT_SPEAKER_COLUMN, "--speaker-column"),
    naming_column: str | None = typer.Option(
        None,
        "--naming-column",
        help="Target-media column used to mirror latent filenames; auto-detects media_path/video by default.",
    ),
    model_name: str = typer.Option(DEFAULT_MODEL, "--model"),
    device: str = typer.Option("cuda", "--device"),
    max_reference_seconds: float = typer.Option(12.0, "--max-reference-seconds", min=1.0),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Compute one normalized speaker embedding for every dataset row."""
    dataset_file = dataset_file.expanduser().resolve()
    rows = _load_rows(dataset_file)
    naming_column = _media_column(rows, naming_column)
    base_dir = dataset_file.parent
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (base_dir / ".precomputed" / "speaker_embeddings").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [i for i, row in enumerate(rows) if speaker_column not in row]
    if missing:
        raise KeyError(f"{len(missing)} dataset rows are missing '{speaker_column}' (first indices: {missing[:5]})")

    torch_device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    console.print(f"Loading speaker encoder [cyan]{model_name}[/] on [cyan]{torch_device}[/]...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).eval().to(torch_device)

    written = skipped = failed = 0
    for row in track(rows, description="Computing speaker embeddings"):
        target_value = str(row[naming_column])
        ref_path = _resolve_path(str(row[speaker_column]), base_dir)
        rel = _output_relative(target_value, base_dir).with_suffix(".pt")
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            waveform = _load_reference_audio(ref_path, max_reference_seconds)
            embedding = _embed(waveform, feature_extractor, model, torch_device)
            torch.save(
                {
                    "embedding": embedding,
                    "encoder": model_name,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "reference_path": str(row[speaker_column]),
                },
                out_path,
            )
            written += 1
        except Exception as exc:  # keep preprocessing long datasets usable
            console.print(f"[red]Failed[/] {ref_path}: {exc}")
            failed += 1

    console.print(
        f"[green]Speaker embeddings complete[/]: written={written}, skipped={skipped}, failed={failed}, "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    app()
