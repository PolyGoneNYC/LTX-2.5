#!/usr/bin/env python3
"""Enroll a character face and route recognized faces to saved LTX speaker embeddings.

POC runtime flow:
    reference image -> face embedding -> character match -> WavLM speaker embedding

The matched speaker embedding is the same 512-D vector consumed by the experimental
LTX-2.5 speaker adapter. Face recognition only selects WHICH saved voice identity to
use; it does not infer a voice from a face.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import torch
import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
DEFAULT_THRESHOLD = 0.363


def _ensure_model(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


def _face_embedding(image_path: Path, model_dir: Path) -> torch.Tensor:
    detector_path = _ensure_model(model_dir / "face_detection_yunet_2023mar.onnx", YUNET_URL)
    recognizer_path = _ensure_model(model_dir / "face_recognition_sface_2021dec.onnx", SFACE_URL)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (width, height), 0.9, 0.3, 5000)
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise ValueError(f"No face detected in {image_path}")

    # Prefer the largest face when a frame contains more than one person.
    face = max(faces, key=lambda row: float(row[2] * row[3]))
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned).reshape(-1)
    embedding = torch.from_numpy(feature).float()
    return torch.nn.functional.normalize(embedding, dim=0).contiguous()


def _speaker_embedding(path: Path) -> tuple[torch.Tensor, str | None]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    embedding = data["embedding"] if isinstance(data, dict) else data
    embedding = torch.as_tensor(embedding).float().reshape(-1)
    embedding = torch.nn.functional.normalize(embedding, dim=0).contiguous()
    encoder = data.get("encoder") if isinstance(data, dict) else None
    return embedding, encoder


@app.command()
def enroll(
    character_id: str,
    face_image: Path = typer.Option(..., "--face", exists=True, dir_okay=False),  # noqa: B008
    speaker_embedding: Path = typer.Option(..., "--speaker", exists=True, dir_okay=False),  # noqa: B008
    registry_dir: Path = typer.Option(Path("character_profiles"), "--registry"),  # noqa: B008
    model_dir: Path = typer.Option(Path(".models/face_router"), "--models"),  # noqa: B008
) -> None:
    """Enroll one face identity and attach its saved 512-D speaker embedding."""
    face_vector = _face_embedding(face_image.resolve(), model_dir.resolve())
    voice_vector, voice_encoder = _speaker_embedding(speaker_embedding.resolve())
    if voice_vector.numel() != 512:
        raise ValueError(f"Expected 512-D speaker embedding, got {voice_vector.numel()}")

    registry_dir = registry_dir.resolve()
    registry_dir.mkdir(parents=True, exist_ok=True)
    output = registry_dir / f"{character_id}.pt"
    torch.save(
        {
            "character_id": character_id,
            "face_embedding": face_vector,
            "speaker_embedding": voice_vector,
            "speaker_encoder": voice_encoder,
            "face_source": str(face_image),
        },
        output,
    )
    print(f"ENROLLED: {character_id}")
    print(f"PROFILE: {output}")
    print(f"FACE DIM: {face_vector.numel()}  VOICE DIM: {voice_vector.numel()}")


@app.command()
def identify(
    face_image: Path = typer.Argument(..., exists=True, dir_okay=False),  # noqa: B008
    registry_dir: Path = typer.Option(Path("character_profiles"), "--registry"),  # noqa: B008
    model_dir: Path = typer.Option(Path(".models/face_router"), "--models"),  # noqa: B008
    threshold: float = typer.Option(DEFAULT_THRESHOLD, "--threshold"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Identify the face and resolve the matching character's speaker embedding."""
    query = _face_embedding(face_image.resolve(), model_dir.resolve())
    profiles = sorted(registry_dir.resolve().glob("*.pt"))
    if not profiles:
        raise ValueError(f"No character profiles found in {registry_dir.resolve()}")

    best_score = -1.0
    best_profile = None
    for profile_path in profiles:
        profile = torch.load(profile_path, map_location="cpu", weights_only=True)
        enrolled = torch.as_tensor(profile["face_embedding"]).float().reshape(-1)
        enrolled = torch.nn.functional.normalize(enrolled, dim=0)
        if enrolled.numel() != query.numel():
            continue
        score = float(torch.dot(query, enrolled))
        if score > best_score:
            best_score = score
            best_profile = profile

    if best_profile is None or best_score < threshold:
        print(f"NO MATCH: best_score={best_score:.4f} threshold={threshold:.4f}")
        raise typer.Exit(code=2)

    character_id = str(best_profile["character_id"])
    voice = torch.as_tensor(best_profile["speaker_embedding"]).float().reshape(-1)
    print(f"MATCH: {character_id}")
    print(f"FACE SCORE: {best_score:.4f}")
    print(f"VOICE DIM: {voice.numel()}")

    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "character_id": character_id,
                "face_score": best_score,
                "speaker_embedding": voice.contiguous(),
                "speaker_encoder": best_profile.get("speaker_encoder"),
            },
            output,
        )
        print(f"ROUTED VOICE: {output}")


if __name__ == "__main__":
    app()
