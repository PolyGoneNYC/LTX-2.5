"""Character voice conditioning primitives for LTX audio diffusion.

The speaker encoder is intentionally external to this module. A caller supplies
an embedding extracted from a short reference recording, and LTX carries that
identity through the denoising path while remaining free to generate dialogue,
ambience and room acoustics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CharacterVoiceCondition:
    """Persistent speaker identity for one named character."""

    character_id: str
    embedding: torch.Tensor
    strength: float = 1.0

    def normalized_embedding(self) -> torch.Tensor:
        """Return a stable unit-length speaker embedding."""
        embedding = self.embedding.float()
        return torch.nn.functional.normalize(embedding, dim=-1)
