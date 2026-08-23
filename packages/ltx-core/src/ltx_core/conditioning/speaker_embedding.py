"""Utilities for injecting persistent speaker identity into LTX audio hidden states.

This is deliberately separate from waveform/reference-latent conditioning.  A
speaker encoder should extract a compact identity embedding from a short WAV;
this module projects that embedding to the transformer's audio hidden width and
adds it to audio tokens during denoising.  The normal LTX audio/video attention
and decoder remain responsible for dialogue timing, ambience, distance and room
acoustics.
"""

from __future__ import annotations

import torch

from ltx_core.conditioning.character_voice import CharacterVoiceCondition


class SpeakerEmbeddingConditioner(torch.nn.Module):
    """Project a speaker embedding and inject it into audio transformer tokens."""

    def __init__(self, speaker_embedding_dim: int, audio_hidden_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(speaker_embedding_dim, audio_hidden_dim, bias=False)
        # Zero-init makes enabling this module an exact no-op until the projection
        # is trained or explicitly initialized from a learned adapter checkpoint.
        torch.nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        condition: CharacterVoiceCondition | None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is None or condition.strength == 0:
            return audio_hidden_states

        speaker = condition.normalized_embedding().to(
            device=audio_hidden_states.device,
            dtype=audio_hidden_states.dtype,
        )
        if speaker.ndim == 1:
            speaker = speaker.unsqueeze(0)

        speaker = self.projection(speaker).unsqueeze(1)
        if speaker.shape[0] == 1 and audio_hidden_states.shape[0] != 1:
            speaker = speaker.expand(audio_hidden_states.shape[0], -1, -1)

        bias = speaker * condition.strength
        if token_mask is not None:
            mask = token_mask.to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            bias = bias * mask

        return audio_hidden_states + bias
