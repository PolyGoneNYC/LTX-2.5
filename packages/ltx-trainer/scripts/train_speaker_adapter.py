#!/usr/bin/env python
"""Train an LTX-2.5 in-diffusion speaker-identity adapter.

Frozen-base architecture:

    WavLM speaker x-vector (512-D)
        -> SpeakerEmbeddingConditioner (~1M trainable params)
        -> global speaker bias on LTX audio tokens
        -> normal LTX audio self-attention / text attention
        -> native LTX audio generation

The 19B LTX transformer stays frozen. Only the small speaker projection is
optimized. The default ``global`` mode is deliberately compatible with real
inference: the speaker identity is available before generation, while future
speech timestamps are not. LTX therefore learns that the identity vector affects
spoken voice while its frozen base model remains responsible for timing, music,
ambience, distance and room acoustics.

Audio-only T2A training is supported and is the recommended first path because
speaker identity does not need to relearn LTX's existing video/lip-sync ability.
Joint AV training remains supported as well.

Dataset requirements
--------------------
Preprocessing must provide:
    audio_latents/
    conditions/
    speaker_embeddings/

Optional legacy/diagnostic mode ``target_speech`` additionally requires:
    speaker_masks/
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader

from ltx_core.conditioning.character_voice import CharacterVoiceCondition
from ltx_core.conditioning.speaker_embedding import SpeakerEmbeddingConditioner
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.trainer import LtxvTrainer, TrainingStepOutput


console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

DEFAULT_SPEAKER_EMBEDDING_DIM = 512
ADAPTER_MODULE_NAME = "speaker_embedding_conditioner"
MASK_MODES = ("global", "target_speech")


class SpeakerAdapterTrainer(LtxvTrainer):
    """LTX trainer variant that updates only speaker conditioning."""

    def __init__(
        self,
        trainer_config: LtxTrainerConfig,
        speaker_embedding_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM,
        speaker_mask_mode: str = "global",
    ) -> None:
        if speaker_mask_mode not in MASK_MODES:
            raise ValueError(f"speaker_mask_mode must be one of {MASK_MODES}, got {speaker_mask_mode!r}")
        self._speaker_embedding_dim = speaker_embedding_dim
        self._speaker_mask_mode = speaker_mask_mode
        super().__init__(trainer_config)

    def _load_models(self) -> None:
        super()._load_models()
        if not self._transformer.model_type.is_audio_enabled():
            raise ValueError("Speaker-adapter training requires an audio-enabled LTX checkpoint")

        adapter = SpeakerEmbeddingConditioner(
            speaker_embedding_dim=self._speaker_embedding_dim,
            audio_hidden_dim=self._transformer.audio_inner_dim,
        )
        setattr(self._transformer, ADAPTER_MODULE_NAME, adapter)
        logger.info(
            "Attached speaker adapter: %d -> %d (%d trainable parameters), mask_mode=%s",
            self._speaker_embedding_dim,
            self._transformer.audio_inner_dim,
            sum(p.numel() for p in adapter.parameters()),
            self._speaker_mask_mode,
        )

    def _collect_trainable_params(self) -> None:
        # Do not install LoRA even when model.training_mode is "lora". That config
        # value is retained only so the stock low-VRAM/quantization path stays available.
        self._transformer.requires_grad_(False)
        adapter: SpeakerEmbeddingConditioner = getattr(self._transformer, ADAPTER_MODULE_NAME)
        adapter.requires_grad_(True)
        self._trainable_params = [p for p in adapter.parameters() if p.requires_grad]
        logger.info("Speaker-adapter trainable params: %d", sum(p.numel() for p in self._trainable_params))

    def _init_dataloader(self) -> None:
        if self._dataset is None:
            data_sources = self._config.training_strategy.get_data_sources().copy()
            if "audio_latents" not in data_sources.values():
                raise ValueError(
                    "Speaker-adapter training needs an audio modality. For the recommended audio-only path use a flexible "
                    "strategy with audio.is_generated=true and audio.latents_dir='audio_latents'."
                )
            data_sources["speaker_embeddings"] = "speaker_embeddings"
            if self._speaker_mask_mode == "target_speech":
                data_sources["speaker_masks"] = "speaker_masks"
            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
            )
            logger.info(
                "Loaded speaker-adapter dataset with %d samples from sources: %s",
                len(self._dataset),
                list(data_sources),
            )

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(dataloader)

    @staticmethod
    def _validate_speaker_mask(mask: Tensor, audio_token_count: int, batch_size: int) -> Tensor:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2:
            raise ValueError(f"speaker_masks/mask must collate to [B,T], got {tuple(mask.shape)}")
        if mask.shape[0] != batch_size:
            raise ValueError(f"speaker mask batch {mask.shape[0]} != audio batch {batch_size}")
        if mask.shape[1] != audio_token_count:
            raise ValueError(
                f"speaker mask token count {mask.shape[1]} != LTX audio token count {audio_token_count}; "
                "regenerate speaker_masks from the same audio_latents used for this training run"
            )
        return mask.float().clamp_(0.0, 1.0)

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        # Preserve the stock LTX text-conditioning path.
        conditions = batch["conditions"]
        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        attention = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(attention, video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
            video_features,
            audio_features,
            additive_mask,
        )
        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)
        if model_inputs.audio is None:
            raise ValueError("Training strategy produced no audio modality")

        embedding_data = batch["speaker_embeddings"]
        speaker_embedding = embedding_data["embedding"]
        if speaker_embedding.ndim != 2:
            raise ValueError(
                "speaker_embeddings/embedding must collate to [B,D], "
                f"got {tuple(speaker_embedding.shape)}"
            )

        batch_size, audio_token_count = model_inputs.audio.latent.shape[:2]
        speaker_mask = None
        if self._speaker_mask_mode == "target_speech":
            speaker_mask = self._validate_speaker_mask(
                batch["speaker_masks"]["mask"],
                audio_token_count=audio_token_count,
                batch_size=batch_size,
            ).to(device=model_inputs.audio.latent.device, dtype=model_inputs.audio.latent.dtype)

        unwrapped = self._accelerator.unwrap_model(self._transformer)
        adapter: SpeakerEmbeddingConditioner = getattr(unwrapped, ADAPTER_MODULE_NAME)
        condition = CharacterVoiceCondition(
            character_id="batch_speaker",
            embedding=speaker_embedding,
            strength=1.0,
        )
        speaker_bias = adapter.project_condition(condition, batch_size=batch_size)

        model_inputs.audio = replace(
            model_inputs.audio,
            speaker_bias=speaker_bias,
            speaker_mask=speaker_mask,
        )

        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)
        sigma = (
            model_inputs.video.sigma.detach()
            if model_inputs.video is not None and model_inputs.video.enabled
            else model_inputs.audio.sigma.detach()
        )
        return TrainingStepOutput(loss=loss, sigma=sigma)

    def _load_checkpoint(self) -> None:
        checkpoint = self._config.model.load_checkpoint
        if not checkpoint:
            self._resume_state = (0, None)
            return

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Speaker adapter checkpoint not found: {checkpoint_path}")

        state = load_file(str(checkpoint_path))
        prefix = "speaker_adapter."
        adapter_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not adapter_state:
            raise ValueError(f"No '{prefix}*' tensors found in {checkpoint_path}")
        adapter: SpeakerEmbeddingConditioner = getattr(self._transformer, ADAPTER_MODULE_NAME)
        adapter.load_state_dict(adapter_state, strict=True)
        self._loaded_checkpoint_path = checkpoint_path
        self._resume_state = (0, None)
        logger.info("Loaded speaker adapter from %s", checkpoint_path)

    def _save_checkpoint(self) -> Path:
        """Save only the speaker adapter, never the frozen 19B base model."""
        output_dir = Path(self._config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        step = max(self._global_step, 0)
        out_path = output_dir / f"speaker_adapter_{step:08d}.safetensors"

        unwrapped = self._accelerator.unwrap_model(self._transformer)
        adapter: SpeakerEmbeddingConditioner = getattr(unwrapped, ADAPTER_MODULE_NAME)
        state = {f"speaker_adapter.{k}": v.detach().float().cpu() for k, v in adapter.state_dict().items()}
        save_file(
            state,
            str(out_path),
            metadata={
                "format": "ltx-2.5-speaker-adapter-v2",
                "speaker_embedding_dim": str(adapter.speaker_embedding_dim),
                "audio_hidden_dim": str(adapter.audio_hidden_dim),
                "speaker_encoder": "microsoft/wavlm-base-plus-sv",
                "speaker_mask": self._speaker_mask_mode,
                "injection": "audio projected tokens before transformer attention",
                "supports_audio_only_training": "true",
            },
        )
        logger.info("Saved speaker adapter to %s", out_path)
        return out_path


@app.command()
def main(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),  # noqa: B008
    speaker_embedding_dim: int = typer.Option(
        DEFAULT_SPEAKER_EMBEDDING_DIM,
        "--speaker-embedding-dim",
        help="512 for microsoft/wavlm-base-plus-sv",
    ),
    speaker_mask_mode: str = typer.Option(
        "global",
        "--speaker-mask-mode",
        help="global (recommended for real inference) or target_speech (diagnostic/training-only timing mask)",
    ),
    disable_progress_bars: bool = typer.Option(False, "--disable-progress-bars"),
) -> None:
    """Train only the LTX-2.5 speaker-conditioning projection."""
    if speaker_mask_mode not in MASK_MODES:
        raise typer.BadParameter(f"must be one of {MASK_MODES}", param_hint="--speaker-mask-mode")

    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    trainer_config = LtxTrainerConfig(**config_data)

    if trainer_config.validation.interval:
        console.print(
            "[yellow]Warning:[/] standard validation does not yet inject speaker embeddings. "
            "Set validation.interval: null for the first adapter run."
        )

    trainer = SpeakerAdapterTrainer(
        trainer_config,
        speaker_embedding_dim=speaker_embedding_dim,
        speaker_mask_mode=speaker_mask_mode,
    )
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
