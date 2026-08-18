"""Text/image-to-video generation with an explicit background + character scene.

Wraps :class:`~ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline`. Instead of describing
the background and characters in the prompt alone, a background plate and one or more character
cutouts are composited into a single concrete frame-0 image (see
:mod:`ltx_pipelines.utils.scene_compositing`), which is then used as strength-1.0 image
conditioning. Anchoring each character's actual pixels at frame 0 keeps their appearance from
drifting over the course of the clip, which text-only prompting cannot guarantee.
"""

import argparse
import logging
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    add_generated_keyframes_arg,
    default_2_stage_arg_parser,
    resolve_cli_params,
    resolve_existing_path,
)
from ltx_pipelines.utils.media_io import encode_video, resolve_hdr_color_space, vae_dtype_for_hdr
from ltx_pipelines.utils.scene_compositing import (
    DEFAULT_CHARACTER_SCALE,
    CharacterPlacement,
    compose_character_scene,
    evenly_spaced_x_fracs,
)

logger = logging.getLogger(__name__)


class CharacterAction(argparse.Action):
    """Parse ``--character PATH [X_FRAC] [SCALE]``.

    PATH is required. X_FRAC (0-1, horizontal center of the character) and SCALE (character
    height as a fraction of canvas height) are optional; an omitted X_FRAC is auto-spaced evenly
    across the frame once every ``--character`` flag has been parsed, and SCALE defaults to
    ``DEFAULT_CHARACTER_SCALE``.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (1, 2, 3):
            msg = f"{option_string} requires 1 to 3 arguments (PATH [X_FRAC] [SCALE]), got {len(values)}"
            raise argparse.ArgumentError(self, msg)
        path = resolve_existing_path(values[0])
        x_frac = float(values[1]) if len(values) > 1 else None
        scale = float(values[2]) if len(values) > 2 else DEFAULT_CHARACTER_SCALE
        current = getattr(namespace, self.dest) or []
        current.append((path, x_frac, scale))
        setattr(namespace, self.dest, current)


def _place_characters(characters: list[tuple[str, float | None, float]]) -> list[CharacterPlacement]:
    """Fill in auto-spaced x_fracs for characters that didn't specify one."""
    auto_fracs = iter(evenly_spaced_x_fracs(sum(1 for _, x_frac, _ in characters if x_frac is None)))
    return [
        CharacterPlacement(path=path, x_frac=x_frac if x_frac is not None else next(auto_fracs), scale=scale)
        for path, x_frac, scale in characters
    ]


def _scene_arg_parser() -> argparse.ArgumentParser:
    params = resolve_cli_params()
    parser = add_generated_keyframes_arg(default_2_stage_arg_parser(params=params, supports_auto_duration=True))
    parser.add_argument(
        "--background",
        type=resolve_existing_path,
        required=True,
        help="Background plate (PNG/JPEG) the characters are composited onto for frame 0.",
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
            "PATH should ideally be a transparent PNG cutout (best identity fidelity); an opaque "
            "photo is also accepted and gets a soft feathered edge instead. X_FRAC (0-1) is the "
            "character's horizontal center, auto-spaced evenly across the frame if omitted. SCALE "
            f"is the character's height as a fraction of the frame height (default {DEFAULT_CHARACTER_SCALE}). "
            "Pass twice for a two-character scene. Example: "
            "--character alice_cutout.png 0.33 0.8 --character bob_cutout.png 0.66 0.75"
        ),
    )
    parser.add_argument(
        "--scene-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for the composited frame-0 scene image (default: 1.0).",
    )
    parser.add_argument(
        "--scene-output",
        type=str,
        default=None,
        help="Where to save the composited scene image, for inspection. Defaults next to --output-path.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _scene_arg_parser()
    args = parser.parse_args()

    scene_output = args.scene_output or str(Path(args.output_path).with_suffix("")) + ".scene.png"
    compose_character_scene(
        background_path=args.background,
        characters=_place_characters(args.characters),
        width=args.width,
        height=args.height,
        output_path=scene_output,
    )
    logger.info("Composited scene image written to %s", scene_output)

    images = [
        ImageConditioningInput(path=scene_output, frame_idx=0, strength=args.scene_strength),
        *args.images,
    ]

    pipeline = TI2VidTwoStagesPipeline(
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
