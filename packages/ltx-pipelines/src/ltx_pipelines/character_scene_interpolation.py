"""Text/image-to-video generation with character scenes anchored at *both* the first and last frame.

Wraps :class:`~ltx_pipelines.keyframe_interpolation.KeyframeInterpolationPipeline`, which
conditions every image (start and end alike) through the guiding-latent keyframe path -- the
mechanism this codebase's own start/end interpolation pipeline uses, rather than the harder
single-frame latent-replacement path that only ever targets frame 0 elsewhere. That matters here:
without an end anchor, only frame 0 is locked to your reference pixels and the rest of the clip,
including the final frame, is free-generated and can drift (a character can visibly change, e.g.
losing their hair, by the last frame). Anchoring the *same* character composite at both ends gives
the model a target to return to, which is the validated way this codebase keeps a subject
recognizable across a whole clip.

See :mod:`ltx_pipelines.character_scene_i2vid` for the single-anchor (start-only) version and
:mod:`ltx_pipelines.utils.scene_compositing` for how the composite images are built.
"""

import argparse
import logging
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.character_scene_i2vid import CharacterAction, _place_characters
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_arg_parser,
    resolve_cli_params,
    resolve_existing_path,
)
from ltx_pipelines.utils.media_io import encode_video, resolve_hdr_color_space, vae_dtype_for_hdr
from ltx_pipelines.utils.scene_compositing import compose_character_scene

logger = logging.getLogger(__name__)


def _scene_arg_parser() -> argparse.ArgumentParser:
    params = resolve_cli_params()
    parser = default_2_stage_arg_parser(params=params)
    parser.add_argument(
        "--background",
        type=resolve_existing_path,
        required=True,
        help="Background plate the start-frame characters are composited onto.",
    )
    parser.add_argument(
        "--character",
        dest="characters",
        action=CharacterAction,
        nargs="+",
        metavar="ARG",
        default=[],
        required=True,
        help="Start-frame character cutout: PATH [X_FRAC] [SCALE]. Pass twice for two characters.",
    )
    parser.add_argument(
        "--end-background",
        type=resolve_existing_path,
        default=None,
        help=(
            "Background plate for the last frame's character composite. Defaults to the same "
            "composited image used for the start frame, so the characters are anchored to be "
            "identical at both ends of the clip."
        ),
    )
    parser.add_argument(
        "--end-character",
        dest="end_characters",
        action=CharacterAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help=(
            "End-frame character cutout: PATH [X_FRAC] [SCALE]. Only used with --end-background; "
            "defaults to reusing --character's cutouts and placements on the new background."
        ),
    )
    parser.add_argument(
        "--scene-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for both composited anchor frames (default: 1.0).",
    )
    parser.add_argument(
        "--scene-output",
        type=str,
        default=None,
        help="Path prefix for the composited scene images, for inspection. Defaults next to --output-path.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _scene_arg_parser()
    args = parser.parse_args()

    scene_prefix = args.scene_output or str(Path(args.output_path).with_suffix(""))
    start_scene = f"{scene_prefix}.start.png"
    compose_character_scene(
        background_path=args.background,
        characters=_place_characters(args.characters),
        width=args.width,
        height=args.height,
        output_path=start_scene,
    )
    logger.info("Composited start scene written to %s", start_scene)

    if args.end_background is None:
        end_scene = start_scene
    else:
        end_scene = f"{scene_prefix}.end.png"
        compose_character_scene(
            background_path=args.end_background,
            characters=_place_characters(args.end_characters or args.characters),
            width=args.width,
            height=args.height,
            output_path=end_scene,
        )
        logger.info("Composited end scene written to %s", end_scene)

    images = [
        ImageConditioningInput(path=start_scene, frame_idx=0, strength=args.scene_strength),
        ImageConditioningInput(path=end_scene, frame_idx=args.num_frames - 1, strength=args.scene_strength),
        *args.images,
    ]

    pipeline = KeyframeInterpolationPipeline(
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
    video, audio, tiling_config = pipeline(
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
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        tiling_config=AUTO_TILING,
        max_batch_size=args.max_batch_size,
    )
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        color_space=hdr,
    )


if __name__ == "__main__":
    main()
