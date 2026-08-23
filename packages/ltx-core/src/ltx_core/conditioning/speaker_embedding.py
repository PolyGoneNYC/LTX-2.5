"""Utilities for injecting persistent speaker identity into LTX audio hidden states.

This is deliberately separate from waveform/reference-latent conditioning. A
speaker encoder should extract a compact identity embedding from a short WAV;
this module projects that embedding to the transformer's audio hidden width and
adds it to audio tokens during denoising. The normal LTX audio/video attention
and decoder remain responsible for dialogue timing, ambience, distance and room
acoustics.
"""

from __future__ import annotations

import torch

from ltx_core.conditioning.character_voice import CharacterVoiceCondition


class SpeakerEmbeddingConditioner(torch.nn.Module):
    """Project a speaker embedding into LTX audio-transformer hidden space.

    The projection is intentionally zero-initialized. That keeps a stock LTX-2.5
    checkpoint bit-for-bit unchanged until this small adapter is trained, while
    still allowing gradients to train the projection from the first step.
    """

    def __init__(self, speaker_embedding_dim: int, audio_hidden_dim: int) -> None:
        super().__init__()
        self.speaker_embedding_dim = speaker_embedding_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.projection = torch.nn.Linear(speaker_embedding_dim, audio_hidden_dim, bias=False)
        torch.nn.init.zeros_(self.projection.weight)

    def project_condition(
        self,
        condition: CharacterVoiceCondition | None,
        batch_size: int | None = None,
    ) -> torch.Tensor | None:
        """Return a pre-projected ``speaker_bias`` suitable for ``Modality``.

        The returned tensor has shape ``(B, 1, D_audio_hidden)``. A batch size of
        one is retained by default so the transformer's normal broadcasting can
        apply the same enrolled speaker to every sample in a batch.
        """
        if condition is None or condition.strength == 0:
            return None

        speaker = condition.normalized_embedding().to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        if speaker.ndim == 1:
            speaker = speaker.unsqueeze(0)
        if speaker.ndim != 2:
            raise ValueError(
                "CharacterVoiceCondition.embedding must be shaped (D,) or (B,D); "
                f"got {tuple(speaker.shape)}"
            )
        if speaker.shape[-1] != self.speaker_embedding_dim:
            raise ValueError(
                f"speaker embedding dimension {speaker.shape[-1]} does not match adapter input "
                f"dimension {self.speaker_embedding_dim}"
            )

        bias = self.projection(speaker).unsqueeze(1) * condition.strength
        if batch_size is not None:
            if bias.shape[0] == 1:
                bias = bias.expand(batch_size, -1, -1)
            elif bias.shape[0] != batch_size:
                raise ValueError(
                    f"speaker-condition batch {bias.shape[0]} does not match requested batch {batch_size}"
                )
        return bias

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        condition: CharacterVoiceCondition | None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible convenience path for direct hidden-state injection."""
        bias = self.project_condition(condition, batch_size=audio_hidden_states.shape[0])
        if bias is None:
            return audio_hidden_states

        bias = bias.to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
        if token_mask is not None:
            mask = token_mask.to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
            if mask.ndim == 1:
                mask = mask.view(1, -1, 1)
            elif mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            elif mask.ndim != 3:
                raise ValueError(f"token_mask must have 1, 2 or 3 dimensions, got {tuple(mask.shape)}")
            bias = bias * mask

        return audio_hidden_states + bias
