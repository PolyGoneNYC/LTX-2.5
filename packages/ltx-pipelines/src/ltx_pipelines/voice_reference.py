"""Reference-voice helpers for LTX audio diffusion experiments.

This intentionally uses LTX's native audio VAE/reference-latent conditioning
instead of replacing a rendered waveform.  The reference remains available to
the diffusion model while scene acoustics and ambience are generated normally.
"""

from __future__ import annotations

import torch

from ltx_core.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
from ltx_core.model.audio_vae import decode_audio_from_file, vae_encode_audio


def build_reference_voice_condition(
    audio_path: str,
    audio_vae,
    audio_patchifier,
    device: torch.device,
    dtype: torch.dtype,
    strength: float = 1.0,
) -> AudioConditionByReferenceLatent:
    """Encode a short WAV/reference clip into native LTX reference tokens.

    This is a proof-of-concept voice-conditioning path.  It preserves the
    reference inside diffusion; it does not perform post-render voice swapping.
    """
    waveform, sample_rate = decode_audio_from_file(audio_path)
    waveform = waveform.to(device=device, dtype=dtype)
    latent = vae_encode_audio(audio_vae, waveform, sample_rate)
    patchified = audio_patchifier.patchify(latent)
    positions = audio_patchifier.get_patch_grid_bounds(latent.shape, device=device)

    # Put the reference immediately before target time, matching DubIt's native
    # reference-audio strategy so target audio can attend to clean voice tokens.
    if positions.numel() > 0:
        duration = positions[:, :, -1, 1].max().item()
        positions = positions - duration - 0.04

    return AudioConditionByReferenceLatent(
        patchified=patchified,
        positions=positions,
        strength=strength,
    )
