"""ComfyUI runtime node for LTX-2.5 face -> character -> native speaker conditioning.

Install by copying this file to ComfyUI/custom_nodes/ltxv_face_voice_adapter.py.

Flow:
    reference IMAGE
        -> OpenCV SFace identity match against saved character_profiles/*.pt
        -> saved 512-D WavLM speaker embedding
        -> trained speaker_adapter_*.safetensors projection
        -> additive bias immediately after LTX audio_patchify_proj
        -> normal frozen LTX transformer denoising

This is NOT post-dubbing. The speaker bias is present inside the LTX denoising pass,
so LTX still generates speech, music, ambience, reverb and A/V timing itself.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
DEFAULT_THRESHOLD = 0.363


def _ensure_model(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


def _face_embedding_from_comfy(image: torch.Tensor, model_dir: Path) -> torch.Tensor:
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Expected ComfyUI IMAGE [B,H,W,C], got {tuple(image.shape)}")

    rgb = image[0, :, :, :3].detach().float().clamp(0, 1).cpu().numpy()
    bgr = cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]

    detector_path = _ensure_model(model_dir / "face_detection_yunet_2023mar.onnx", YUNET_URL)
    recognizer_path = _ensure_model(model_dir / "face_recognition_sface_2021dec.onnx", SFACE_URL)
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (width, height), 0.9, 0.3, 5000)
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        raise ValueError("No face detected in reference image")

    face = max(faces, key=lambda row: float(row[2] * row[3]))
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    aligned = recognizer.alignCrop(bgr, face)
    feature = recognizer.feature(aligned).reshape(-1)
    embedding = torch.from_numpy(feature).float()
    return torch.nn.functional.normalize(embedding, dim=0).contiguous()


def _match_profile(query: torch.Tensor, registry_dir: Path, threshold: float) -> tuple[dict, float]:
    profiles = sorted(registry_dir.glob("*.pt"))
    if not profiles:
        raise ValueError(f"No character profiles found in {registry_dir}")

    best_profile = None
    best_score = -1.0
    for path in profiles:
        profile = torch.load(path, map_location="cpu", weights_only=True)
        enrolled = torch.as_tensor(profile["face_embedding"]).float().reshape(-1)
        enrolled = torch.nn.functional.normalize(enrolled, dim=0)
        if enrolled.numel() != query.numel():
            continue
        score = float(torch.dot(query, enrolled))
        if score > best_score:
            best_score = score
            best_profile = profile

    if best_profile is None or best_score < threshold:
        raise ValueError(f"No enrolled character matched face: best={best_score:.4f}, threshold={threshold:.4f}")
    return best_profile, best_score


def _load_projected_bias(adapter_path: Path, speaker_embedding: torch.Tensor, expected_hidden_dim: int) -> torch.Tensor:
    state = load_file(str(adapter_path), device="cpu")
    key = "speaker_adapter.projection.weight"
    if key not in state:
        raise ValueError(f"Adapter does not contain {key}: {adapter_path}")
    weight = state[key].float()
    speaker = torch.nn.functional.normalize(speaker_embedding.float().reshape(-1), dim=0)
    if weight.ndim != 2 or weight.shape[1] != speaker.numel():
        raise ValueError(
            f"Adapter weight {tuple(weight.shape)} is incompatible with speaker vector {speaker.numel()}"
        )
    if weight.shape[0] != expected_hidden_dim:
        raise ValueError(
            f"Adapter output dim {weight.shape[0]} does not match loaded LTX audio hidden dim {expected_hidden_dim}"
        )
    return torch.mv(weight, speaker).contiguous()


class LTXVFaceVoiceAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_image": ("IMAGE",),
                "adapter_file": (
                    "STRING",
                    {"default": "/workspace/runpod-slim/speaker_adapter.safetensors"},
                ),
                "registry_dir": (
                    "STRING",
                    {"default": "/workspace/runpod-slim/character_profiles"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "face_threshold": (
                    "FLOAT",
                    {"default": DEFAULT_THRESHOLD, "min": -1.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "FLOAT")
    RETURN_NAMES = ("model", "character", "face_score")
    FUNCTION = "execute"
    CATEGORY = "conditioning/ltxv"
    DESCRIPTION = (
        "Recognizes an enrolled character from the reference image and injects that character's "
        "trained WavLM speaker identity inside LTX's audio denoising path. No post-dubbing."
    )

    def execute(self, model, reference_image, adapter_file, registry_dir, strength, face_threshold):
        adapter_path = Path(adapter_file).expanduser().resolve()
        registry = Path(registry_dir).expanduser().resolve()
        if not adapter_path.is_file():
            raise ValueError(f"Speaker adapter not found: {adapter_path}")
        if not registry.is_dir():
            raise ValueError(f"Character registry not found: {registry}")

        model_dir = Path(__file__).resolve().parent / ".models" / "face_router"
        query = _face_embedding_from_comfy(reference_image, model_dir)
        profile, score = _match_profile(query, registry, face_threshold)
        character_id = str(profile["character_id"])
        speaker = torch.as_tensor(profile["speaker_embedding"]).float().reshape(-1)

        diffusion_model = model.get_model_object("diffusion_model")
        audio_hidden_dim = int(getattr(diffusion_model, "audio_inner_dim"))
        bias_cpu = _load_projected_bias(adapter_path, speaker, audio_hidden_dim) * float(strength)

        original_process_input = model.get_model_object("diffusion_model._process_input")

        def process_input_with_character_voice(*args, **kwargs):
            hidden_states, positions, additional_args = original_process_input(*args, **kwargs)
            if not isinstance(hidden_states, (list, tuple)) or len(hidden_states) < 2:
                return hidden_states, positions, additional_args

            audio_hidden = hidden_states[1]
            if audio_hidden is None or audio_hidden.shape[1] == 0:
                return hidden_states, positions, additional_args

            # Do not modify optional frozen reference-audio context tokens. The trained
            # speaker identity is applied only to the target audio tokens being denoised.
            ref_len = int(additional_args.get("ref_audio_seq_len", 0) or 0)
            bias = bias_cpu.to(device=audio_hidden.device, dtype=audio_hidden.dtype).view(1, 1, -1)
            audio_out = audio_hidden.clone()
            audio_out[:, ref_len:, :] = audio_out[:, ref_len:, :] + bias

            out = list(hidden_states)
            out[1] = audio_out
            return out, positions, additional_args

        patched = model.clone()
        patched.add_object_patch("diffusion_model._process_input", process_input_with_character_voice)
        return (patched, character_id, score)


NODE_CLASS_MAPPINGS = {
    "LTXVFaceVoiceAdapter": LTXVFaceVoiceAdapter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVFaceVoiceAdapter": "LTXV Face → Character Voice Adapter",
}
