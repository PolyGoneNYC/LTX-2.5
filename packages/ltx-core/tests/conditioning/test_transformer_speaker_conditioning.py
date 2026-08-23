import pytest
import torch

from ltx_core.model.transformer.transformer_args import apply_speaker_conditioning


def test_speaker_bias_broadcasts_over_audio_tokens() -> None:
    hidden = torch.zeros(2, 4, 3)
    bias = torch.tensor([1.0, 2.0, 3.0])

    out = apply_speaker_conditioning(hidden, bias, None)

    expected = bias.view(1, 1, 3).expand_as(hidden)
    assert torch.allclose(out, expected)


def test_speaker_mask_limits_identity_to_speaking_region() -> None:
    hidden = torch.zeros(1, 5, 2)
    bias = torch.tensor([[[2.0, -1.0]]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.25, 0.0]])

    out = apply_speaker_conditioning(hidden, bias, mask)

    assert torch.allclose(out[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(out[0, 1], torch.tensor([2.0, -1.0]))
    assert torch.allclose(out[0, 2], torch.tensor([2.0, -1.0]))
    assert torch.allclose(out[0, 3], torch.tensor([0.5, -0.25]))
    assert torch.allclose(out[0, 4], torch.tensor([0.0, 0.0]))


def test_speaker_bias_rejects_wrong_hidden_width() -> None:
    hidden = torch.zeros(1, 3, 4)
    bias = torch.zeros(1, 1, 5)

    with pytest.raises(ValueError, match="hidden dimension"):
        apply_speaker_conditioning(hidden, bias, None)


def test_speaker_mask_rejects_wrong_token_count() -> None:
    hidden = torch.zeros(1, 3, 4)
    bias = torch.zeros(1, 1, 4)
    mask = torch.ones(1, 2)

    with pytest.raises(ValueError, match="token dimension"):
        apply_speaker_conditioning(hidden, bias, mask)
