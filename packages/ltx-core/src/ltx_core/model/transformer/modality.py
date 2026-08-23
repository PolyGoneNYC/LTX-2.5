from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """Input data for a single modality (video or audio) in the transformer.

    ``speaker_bias`` and ``speaker_mask`` are optional audio-only conditioning
    tensors. ``speaker_bias`` is already projected to the transformer hidden
    width so the pretrained LTX checkpoint shapes remain unchanged.
    ``speaker_mask`` can restrict the identity bias to selected audio tokens.
    """

    latent: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None
    speaker_bias: torch.Tensor | None = None
    speaker_mask: torch.Tensor | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes."""
        n = len(sizes)
        split_fields: dict[str, list[torch.Tensor | None] | list[bool]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]
