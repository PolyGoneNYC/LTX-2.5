import torch

from ltx_core.conditioning.character_voice import CharacterVoiceCondition
from ltx_core.conditioning.speaker_embedding import SpeakerEmbeddingConditioner


def test_no_condition_is_noop() -> None:
    conditioner = SpeakerEmbeddingConditioner(8, 16)
    x = torch.randn(2, 5, 16)
    assert torch.equal(conditioner(x, None), x)


def test_zero_initialized_projection_is_noop() -> None:
    conditioner = SpeakerEmbeddingConditioner(8, 16)
    x = torch.randn(1, 5, 16)
    condition = CharacterVoiceCondition("A", torch.randn(8), strength=1.0)
    assert torch.equal(conditioner(x, condition), x)


def test_condition_can_bias_audio_tokens() -> None:
    conditioner = SpeakerEmbeddingConditioner(2, 3)
    with torch.no_grad():
        conditioner.projection.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
    x = torch.zeros(1, 4, 3)
    condition = CharacterVoiceCondition("A", torch.tensor([1.0, 0.0]), strength=0.5)
    out = conditioner(x, condition)
    expected = torch.tensor([0.5, 0.0, 0.5]).view(1, 1, 3).expand_as(x)
    assert torch.allclose(out, expected)


def test_token_mask_limits_injection() -> None:
    conditioner = SpeakerEmbeddingConditioner(2, 2)
    with torch.no_grad():
        conditioner.projection.weight.copy_(torch.eye(2))
    x = torch.zeros(1, 3, 2)
    condition = CharacterVoiceCondition("A", torch.tensor([1.0, 0.0]))
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    out = conditioner(x, condition, token_mask=mask)
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(out[0, 1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(out[0, 2], torch.tensor([1.0, 0.0]))
