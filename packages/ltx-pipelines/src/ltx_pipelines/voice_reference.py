"""Reference-voice conditioning helpers for LTX audio diffusion experiments."""

from __future__ import annotations

import torch

from ltx_core.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
from ltx_pipelines.dubit import patchify_dubit_audio_reference_latent


def build_reference_voice_condition_from_latent(
    audio_latent: torch.Tensor,
    device: torch.device,
    strength: float = 1.0,
) -> AudioConditionByReferenceLatent:
    """Turn an LTX audio-VAE latent into clean reference tokens.

    The tokens are placed immediately before target time, matching DubIt's
    native diffusion-time reference-audio strategy.  This deliberately avoids
    post-render waveform replacement so LTX remains responsible for scene
    ambience, distance and acoustics.
    """
    patchified, positions = patchify_dubit_audio_reference_latent(
        audio_latent,
        negative_positions=True,
        device=device,
    )
    return AudioConditionByReferenceLatent(
        patchified,
        positions,
        strength=strength,
    )
