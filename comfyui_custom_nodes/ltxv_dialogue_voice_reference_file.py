"""ComfyUI node: synthesize exact dialogue in a reference voice for LTX-2.5 diffusion conditioning.

This is intentionally NOT a post-processing voice replacement node.

The node takes:
  * exact dialogue text to be spoken, and
  * a path to a short reference WAV for the target speaker.

It uses OmniVoice to synthesize the requested dialogue in that reference voice, then returns
the result as a normal ComfyUI AUDIO object. Feed that AUDIO into ComfyUI's built-in
LTXVReferenceAudio node. LTX then sees the actual target performance while its audio/video
streams are denoised, instead of seeing an unrelated voice sample.

This is the practical no-training LTX-2.5 POC: the reference clip contains BOTH the target
speaker and the exact target words. It avoids the failure mode where a generic 5-second sample
can influence mouth motion/content without strongly preserving speaker identity.

Dependencies (already used by ltxv_voice_lock on the project RunPod):
    faster-whisper
    omnivoice
    numpy
    torch
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_OMNIVOICE_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _load_omnivoice_model(model_path: str, device: str):
    """Load/reuse OmniVoice instead of paying the model load cost on every queue."""
    cache_key = (model_path, device)
    cached = _OMNIVOICE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(model_path, device_map=device, dtype=torch.float16)
    _OMNIVOICE_MODEL_CACHE[cache_key] = model
    return model


def _reference_transcript(path: Path, whisper_model: str) -> str:
    """Transcribe the reference clip once so OmniVoice can build a zero-shot clone prompt."""
    from faster_whisper import WhisperModel

    # CPU is intentional: this is a single short clip and avoids CUDA-runtime conflicts with
    # the main ComfyUI environment.
    whisper = WhisperModel(whisper_model, device="cpu")
    segments, _ = whisper.transcribe(str(path), language="en")
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        raise ValueError(
            "[LTXVDialogueVoiceReferenceFile] Could not transcribe any speech from the reference WAV."
        )
    return text


class LTXVDialogueVoiceReferenceFile:
    """Create the exact requested line in the target voice for LTXVReferenceAudio."""

    CATEGORY = "audio/ltxv"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("reference_audio", "reference_transcript")
    FUNCTION = "synthesize"
    DESCRIPTION = (
        "Synthesizes the exact dialogue in a voice cloned from reference_voice_path. "
        "Connect reference_audio to LTXVReferenceAudio so the cloned performance participates "
        "during LTX generation rather than replacing the finished soundtrack afterward."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What are you looking at?",
                        "tooltip": (
                            "Exact words the character should say. Keep this identical to the "
                            "quoted dialogue in the main LTX prompt."
                        ),
                    },
                ),
                "reference_voice_path": (
                    "STRING",
                    {
                        "default": "/workspace/runpod-slim/ComfyUI/input/reference_voice.wav",
                        "tooltip": "Path to the target speaker's clean reference WAV. About 5-10 seconds is ideal.",
                    },
                ),
                "whisper_model": (
                    "STRING",
                    {
                        "default": "base.en",
                        "tooltip": "Used only to transcribe the reference WAV for OmniVoice cloning.",
                    },
                ),
                "omnivoice_model_path": (
                    "STRING",
                    {"default": "k2-fsa/OmniVoice"},
                ),
                "device": (
                    "STRING",
                    {"default": "cuda:0"},
                ),
            }
        }

    def synthesize(
        self,
        dialogue_text: str,
        reference_voice_path: str,
        whisper_model: str,
        omnivoice_model_path: str,
        device: str,
    ):
        line = dialogue_text.strip()
        if not line:
            raise ValueError("[LTXVDialogueVoiceReferenceFile] dialogue_text is empty.")

        ref_path = Path(reference_voice_path).expanduser()
        if not ref_path.is_file():
            raise ValueError(
                f"[LTXVDialogueVoiceReferenceFile] reference WAV not found: {ref_path}"
            )

        ref_text = _reference_transcript(ref_path, whisper_model)
        print(
            f"[LTXVDialogueVoiceReferenceFile] reference transcript: {ref_text!r}"
        )

        model = _load_omnivoice_model(omnivoice_model_path, device)
        clone_prompt = model.create_voice_clone_prompt(str(ref_path), ref_text=ref_text)

        # No requested duration here: this generated performance becomes the timing reference
        # that LTXVReferenceAudio exposes to the diffusion model. pad_duration=0 avoids an
        # artificial leading/trailing silence that would shift mouth timing.
        audios = model.generate(
            text=line,
            voice_clone_prompt=clone_prompt,
            pad_duration=0.0,
        )
        if not audios:
            raise RuntimeError("[LTXVDialogueVoiceReferenceFile] OmniVoice returned no audio.")

        clip = np.asarray(audios[0], dtype=np.float32).squeeze()
        if clip.ndim != 1:
            # Be conservative if a future OmniVoice build returns channels.
            clip = np.mean(clip, axis=0, dtype=np.float32)

        sample_rate = int(model.sampling_rate)
        duration = len(clip) / sample_rate if sample_rate else 0.0
        print(
            f"[LTXVDialogueVoiceReferenceFile] synthesized {duration:.2f}s "
            f"at {sample_rate} Hz: {line!r}"
        )

        audio = {
            "waveform": torch.from_numpy(clip).reshape(1, 1, -1),
            "sample_rate": sample_rate,
        }
        return (audio, ref_text)


NODE_CLASS_MAPPINGS = {
    "LTXVDialogueVoiceReferenceFile": LTXVDialogueVoiceReferenceFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVDialogueVoiceReferenceFile": "LTXV Dialogue Voice Reference (WAV)",
}
