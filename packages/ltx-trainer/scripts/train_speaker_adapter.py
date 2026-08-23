#!/usr/bin/env python
"""Train the first LTX-2.5 in-diffusion speaker-identity adapter.

This is intentionally a small, frozen-base experiment:

    WavLM speaker x-vector (512-D)
        -> SpeakerEmbeddingConditioner (~1M trainable params)
        -> speaker_bias on LTX audio tokens
        -> normal LTX audio self-attention + A/V cross-attention
        -> native LTX audio/video generation

The 19B LTX transformer stays frozen.  Only the speaker projection is optimized.
That makes the experiment feasible on a 32 GB 5090 with the same base-model
quantization used by low-VRAM LoRA training.

Dataset requirements
--------------------
Use the normal joint A/V preprocessing first (latents + audio_latents +
conditions), then run ``compute_speaker_embeddings.py`` so the precomputed tree
also contains ``speaker_embeddings`` with matching relative .pt paths.

For this first smoke-train the speaker identity is applied to all audio tokens.
The core API already supports ``speaker_mask``; a later dataset pass will add
word/VAD masks so music and ambience tokens can be left completely unconditioned.
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


class SpeakerAdapterTrainer(LtxvTrainer):
    """LTX trainer variant that updates only the speaker-conditioning adapter."""

    def __init__(self, trainer_config: LtxTrainerConfig, speaker_embedding_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM):
        self._speaker_embedding_dim = speaker_embedding_dim
        super().__init__(trainer_config)

    def _load_models(self) -> None:
        # Stock loader may quantize the frozen base model according to the config.
        super()._load_models()

        if not self._transformer.model_type.is_audio_enabled():
            raise ValueError("Speaker-adapter training requires an audio-enabled LTX checkpoint")

        adapter = SpeakerEmbeddingConditioner(
            speaker_embedding_dim=self._speaker_embedding_dim,
            audio_hidden_dim=self._transformer.audio_inner_dim,
        )
        # Attach it to the transformer so Accelerate moves/wraps it together with the
        # frozen LTX model and checkpoint unwrapping has one canonical owner.
        setattr(self._transformer, ADAPTER_MODULE_NAME, adapter)
        logger.info(
            "Attached speaker adapter: %d -> %d (%d trainable parameters)",
            self._speaker_embedding_dim,
            self._transformer.audio_inner_dim,
            sum(p.numel() for p in adapter.parameters()),
        )

    def _collect_trainable_params(self) -> None:
        # Override the normal full/LoRA branches completely. The base 19B model is
        # frozen and there are no PEFT LoRA layers in this experiment.
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
                    "Speaker-adapter training needs joint audio/video data. Set "
                    "training_strategy.name=text_to_video and with_audio=true (or use an "
                    "equivalent audio-enabled strategy)."
                )
            data_sources["speaker_embeddings"] = "speaker_embeddings"
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

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        # Keep the stock text-conditioning path exactly the same as LtxvTrainer.
        conditions = batch["conditions"]
        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
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

        # After Accelerate wrapping, use the unwrapped object only to locate the
        # adapter module; its parameters are the same Parameters owned by the wrapped model.
        unwrapped = self._accelerator.unwrap_model(self._transformer)
        adapter: SpeakerEmbeddingConditioner = getattr(unwrapped, ADAPTER_MODULE_NAME)
        condition = CharacterVoiceCondition(
            character_id="batch_speaker",
            embedding=speaker_embedding,
            strength=1.0,
        )
        speaker_bias = adapter.project_condition(
            condition,
            batch_size=model_inputs.audio.latent.shape[0],
        )
        model_inputs.audio = replace(
            model_inputs.audio,
            speaker_bias=speaker_bias,
            speaker_mask=None,  # Phase 1 smoke train; VAD/token masks are Phase 2.
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
        """Load an adapter-only safetensors checkpoint when configured."""
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
        self._resume_state = (0, None)  # weights-only resume for the research POC
        logger.info("Loaded speaker adapter from %s", checkpoint_path)

    def _save_checkpoint(self) -> Path:
        """Save only the ~1M-parameter speaker adapter, never the frozen 19B base."""
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
                "format": "ltx-2.5-speaker-adapter-v1",
                "speaker_embedding_dim": str(adapter.speaker_embedding_dim),
                "audio_hidden_dim": str(adapter.audio_hidden_dim),
                "speaker_encoder": "microsoft/wavlm-base-plus-sv",
                "injection": "audio projected tokens before transformer attention",
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
    disable_progress_bars: bool = typer.Option(False, "--disable-progress-bars"),
) -> None:
    """Train only the LTX-2.5 speaker-conditioning projection."""
    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    trainer_config = LtxTrainerConfig(**config_data)

    if trainer_config.validation.interval:
        console.print(
            "[yellow]Warning:[/] standard validation does not yet inject speaker embeddings. "
            "Set validation.interval: 0 for the first adapter run."
        )

    trainer = SpeakerAdapterTrainer(
        trainer_config,
        speaker_embedding_dim=speaker_embedding_dim,
    )
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
