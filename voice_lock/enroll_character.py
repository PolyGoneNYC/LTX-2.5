"""Build a character "profile": a face-identity embedding plus a reference voice clip path.

Run this once per character, then pass the resulting JSON profile to lock_character_voice.py for
every generated video that features them.

Usage:
    python enroll_character.py --name farley \
        --reference-photo farley_face1.jpg --reference-photo farley_face2.jpg \
        --reference-voice farley_voice.wav --profile-dir profiles/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def build_face_embedding(face_app: FaceAnalysis, photo_paths: list[str]) -> np.ndarray:
    """Average the normalized embedding of the largest detected face in each reference photo."""
    embeddings = []
    for path in photo_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        faces = face_app.get(img)
        if not faces:
            raise ValueError(f"No face detected in reference photo: {path}")
        # Largest face by bbox area, in case a reference photo has more than one person in it.
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(largest.normed_embedding)
    mean = np.mean(embeddings, axis=0)
    return mean / np.linalg.norm(mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Character name, e.g. 'farley'.")
    parser.add_argument(
        "--reference-photo",
        dest="reference_photos",
        action="append",
        required=True,
        help="Path to a reference photo of the character's face. Repeatable -- more angles/"
        "lighting conditions give a more robust embedding.",
    )
    parser.add_argument(
        "--reference-voice",
        required=True,
        help="Path to a ~5-10s wav clip of the character's target voice, for OpenVoice.",
    )
    parser.add_argument(
        "--profile-dir",
        default="profiles",
        help="Directory to write <name>.json into (default: profiles/).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="InsightFace detector input size (default: 640).",
    )
    args = parser.parse_args()

    for photo in args.reference_photos:
        if not Path(photo).exists():
            raise FileNotFoundError(photo)
    if not Path(args.reference_voice).exists():
        raise FileNotFoundError(args.reference_voice)

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    embedding = build_face_embedding(face_app, args.reference_photos)

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{args.name}.json"
    profile = {
        "name": args.name,
        "face_embedding": embedding.tolist(),
        "reference_voice_path": str(Path(args.reference_voice).resolve()),
        "reference_photos": [str(Path(p).resolve()) for p in args.reference_photos],
    }
    profile_path.write_text(json.dumps(profile, indent=2))
    print(f"Wrote profile for '{args.name}' to {profile_path}")


if __name__ == "__main__":
    main()
