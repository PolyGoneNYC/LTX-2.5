"""Text/image-to-video generation with the background hard-locked for the whole clip.

Wraps a variant of :class:`~ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline`.
`CharacterSceneI2VidPipeline` (and even `CharacterSceneInterpolationPipeline`, which anchors both
endpoints) only lock specific *frames* -- every pixel in every unconditioned frame, background
included, is free-generated and can drift. This pipeline instead locks a *region*: the background
pixels are clamped to the composited scene image for every single frame of the clip (not just
frame 0 or the last frame), via :class:`~ltx_core.conditioning.types.mask_cond.VideoConditionByMask`,
while the character region (auto-derived from the compositing footprint -- see
:mod:`ltx_pipelines.utils.scene_compositing`) is left fully free to move. Nothing forces the
background to change, so it can't drift; the model only has to animate the character.

``VideoConditionByMask`` is an existing, already-used primitive (see
``packages/ltx-trainer/src/ltx_trainer/validation_runner.py``) -- this pipeline is the first place
it's wired into a user-facing generation pipeline rather than training-time validation.
"""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem, VideoConditionByMask
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import (
    AUTO_TILING,
    AutoTiling,
    TileSizeConfig,
    TilingConfig,
    VideoEncoder,
    get_video_chunks_number,
)
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.character_scene_i2vid import CharacterAction, _place_characters
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    add_generated_keyframes_arg,
    default_2_stage_arg_parser,
    resolve_cli_params,
    resolve_existing_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    get_device,
    has_generated_keyframes,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    HDRColorSpace,
    encode_video,
    load_image_and_preprocess,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.scene_compositing import compose_character_scene
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)


def _background_lock_conditioning(  # noqa: PLR0913
    scene_image_path: str,
    mask_image_path: str,
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    strength: float,
    color_space: HDRColorSpace | None,
    crf: int | None,
) -> ConditioningItem:
    """Hard-lock the background for the whole clip.

    Encodes a static video (the composited scene image repeated for every frame) and masks the
    latent so only the background region (the *inverse* of the character footprint mask) is
    clamped to it at every frame; the character region is left with ``denoise_mask=1``, fully
    free to be generated/animated. *crf* should match the frame-0 anchor's resolved CRF
    (``images[0].crf`` after ``ImageConditioner.resolve_crf``), so this reference is recompressed
    the same way as every other conditioning image the checkpoint expects.
    """
    frame = load_image_and_preprocess(
        image_path=scene_image_path,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        crf=crf,
        color_space=color_space,
    )  # (1, C, 1, H, W)
    static_video = frame.expand(-1, -1, num_frames, -1, -1).contiguous()
    latent = video_encoder.tiled_encode(static_video, TileSizeConfig.default())  # (1, C, F', H', W')

    footprint = Image.open(mask_image_path).convert("L").resize((width, height), Image.NEAREST)
    footprint_t = torch.from_numpy(np.array(footprint, dtype=np.float32)).to(device=device)
    footprint_t = footprint_t.reshape(1, 1, height, width) / 255.0
    locked_spatial = 1.0 - footprint_t  # background = inverse of the character footprint

    latent_h, latent_w = latent.shape[-2], latent.shape[-1]
    locked_spatial = F.interpolate(locked_spatial, size=(latent_h, latent_w), mode="nearest")
    locked_spatial = (locked_spatial.squeeze(0).squeeze(0) > 0.5).to(dtype=torch.float32)

    latent_frames = latent.shape[2]
    locked_mask = locked_spatial.unsqueeze(0).unsqueeze(0).expand(1, latent_frames, -1, -1).contiguous()

    return VideoConditionByMask(latent=latent, mask=locked_mask, strength=strength)


class CharacterSceneLockedBackgroundPipeline:
    """
    Two-stage text/image-to-video generation pipeline with the background locked for the whole
    clip. Same stage structure as ``TI2VidTwoStagesPipeline``; see module docstring for what's
    different.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.upsampler = VideoUpsampler(
            model_paths.video_vae(),
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path,
            self.dtype,
            self.device,
        )

        self.stage_1 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage_2 = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        scene_image_path: str,
        scene_mask_path: str,
        background_lock_strength: float = 1.0,
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space: HDRColorSpace | None = None,
        generated_keyframes: int | list[int] = 0,
    ) -> tuple[Iterator[torch.Tensor], Audio, int, TilingConfig | None]:
        require_num_frames_source(num_frames, self.duration_predictor)
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        if has_generated_keyframes(generated_keyframes):
            self.stage_1.assert_generated_keyframes_supported()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=v_context_p,
            audio_encoding=a_context_p,
            frame_rate=frame_rate,
        )

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: [
                *combined_image_conditionings(
                    images=images,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                    color_space=color_space,
                ),
                _background_lock_conditioning(
                    scene_image_path=scene_image_path,
                    mask_image_path=scene_mask_path,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    num_frames=num_frames,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                    strength=background_lock_strength,
                    color_space=color_space,
                    crf=images[0].crf,
                ),
            ]
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_1(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params, negative_context=v_context_n
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params, negative_context=a_context_n
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=max_batch_size,
        )

        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: [
                *combined_image_conditionings(
                    images=images,
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                    color_space=color_space,
                ),
                _background_lock_conditioning(
                    scene_image_path=scene_image_path,
                    mask_image_path=scene_mask_path,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                    strength=background_lock_strength,
                    color_space=color_space,
                    crf=images[0].crf,
                ),
            ]
        )

        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio, num_frames, tiling_config


def _scene_arg_parser() -> argparse.ArgumentParser:
    params = resolve_cli_params()
    parser = add_generated_keyframes_arg(default_2_stage_arg_parser(params=params, supports_auto_duration=True))
    parser.add_argument(
        "--background",
        type=resolve_existing_path,
        required=True,
        help="Background plate the characters are composited onto, and locked in place for the whole clip.",
    )
    parser.add_argument(
        "--character",
        dest="characters",
        action=CharacterAction,
        nargs="+",
        metavar="ARG",
        default=[],
        required=True,
        help=(
            "Character cutout to composite onto the background: PATH [X_FRAC] [SCALE]. "
            "Pass twice for two characters. Everywhere NOT covered by a character becomes the "
            "locked background region."
        ),
    )
    parser.add_argument(
        "--scene-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for the composited frame-0 scene image (default: 1.0).",
    )
    parser.add_argument(
        "--background-lock-strength",
        type=float,
        default=1.0,
        help=(
            "How strongly the background region is clamped to the scene image for every frame "
            "(default: 1.0, fully locked). Lower values loosen the clamp if you want subtle "
            "background motion (e.g. lighting flicker) instead of a perfectly static background."
        ),
    )
    parser.add_argument(
        "--scene-output",
        type=str,
        default=None,
        help="Path prefix for the composited scene image and its footprint mask, for inspection.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _scene_arg_parser()
    args = parser.parse_args()

    scene_prefix = args.scene_output or str(Path(args.output_path).with_suffix(""))
    scene_image = f"{scene_prefix}.scene.png"
    scene_mask = f"{scene_prefix}.scene_mask.png"
    compose_character_scene(
        background_path=args.background,
        characters=_place_characters(args.characters),
        width=args.width,
        height=args.height,
        output_path=scene_image,
        mask_output_path=scene_mask,
    )
    logger.info("Composited scene image written to %s (mask: %s)", scene_image, scene_mask)

    images = [
        ImageConditioningInput(path=scene_image, frame_idx=0, strength=args.scene_strength),
        *args.images,
    ]

    pipeline = CharacterSceneLockedBackgroundPipeline(
        model_paths=args.model_paths,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    hdr = resolve_hdr_color_space(images=images, hdr=args.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    video, audio, num_frames, tiling_config = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=images,
        scene_image_path=scene_image,
        scene_mask_path=scene_mask,
        background_lock_strength=args.background_lock_strength,
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        max_batch_size=args.max_batch_size,
        tiling_config=AUTO_TILING,
        generated_keyframes=args.num_generated_keyframes,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
        color_space=hdr,
    )


if __name__ == "__main__":
    main()
