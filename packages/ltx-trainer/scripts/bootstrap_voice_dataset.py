#!/usr/bin/env python3
"""Build a small audio-only speaker-adapter dataset from one clean reference voice.

Run this with an environment that already has OmniVoice installed (for this project,
the ComfyUI Python environment is normally the easiest). It creates several DIFFERENT
utterances in the cloned voice, so the LTX speaker adapter learns timbre/identity rather
than memorizing one sentence.

Output:
    <output>/reference_voice.wav
    <output>/clips/voice_001.wav ...
    <output>/dataset.json

The resulting dataset is compatible with LTX trainer T2A preprocessing and
compute_speaker_embeddings.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

PHRASES = [
    "What are you looking at?",
    "I told you I'd be back before dinner.",
    "The weather changed faster than anyone expected.",
    "Please close the door and turn the music down.",
    "I don't know why you're laughing, but it's making me laugh too.",
    "We have one more chance to get this right.",
    "Can you hear me clearly from across the room?",
    "I left the keys on the kitchen counter.",
    "That was not part of the plan.",
    "Give me a minute and I'll explain everything.",
    "It's almost midnight, and we're still waiting.",
    "You really thought I wouldn't notice?",
    "The train leaves at seven thirty tomorrow morning.",
    "Take the blue folder and put it beside the lamp.",
    "I can't believe we actually pulled that off.",
    "Everything sounded different once the door was closed.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_voice")
    parser.add_argument("output_dir")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    reference = Path(args.reference_voice).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    if not reference.is_file():
        raise FileNotFoundError(reference)

    copied_reference = output / "reference_voice.wav"
    shutil.copy2(reference, copied_reference)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading OmniVoice {args.model} on {args.device}...")
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=dtype)
    print("Encoding reference voice once...")
    clone_prompt = model.create_voice_clone_prompt(str(copied_reference), ref_text=None)

    rows = []
    count = max(1, min(args.count, len(PHRASES)))
    for i, text in enumerate(PHRASES[:count], start=1):
        print(f"[{i}/{count}] {text}")
        audio = model.generate(
            text=text,
            language="English",
            voice_clone_prompt=clone_prompt,
            pad_duration=0.05,
        )[0]
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        rel = Path("clips") / f"voice_{i:03d}.wav"
        sf.write(output / rel, audio, model.sampling_rate)
        rows.append(
            {
                "audio": rel.as_posix(),
                "caption": f'A woman speaks clearly in a natural conversational voice: "{text}"',
                "speaker_reference": "reference_voice.wav",
            }
        )

    dataset_file = output / "dataset.json"
    dataset_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"DATASET: {dataset_file}")
    print(f"CLIPS: {len(rows)}")
    print("READY FOR LTX T2A PREPROCESSING")


if __name__ == "__main__":
    main()
