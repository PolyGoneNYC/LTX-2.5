"""Seed LTX-2.5's actual target audio latent with cloned speech.

This node is for the case where reference-audio context changes timing/content but does not
reliably preserve speaker timbre on the base LTX-2.5 model. Instead of asking the model to infer
the voice from context, we encode the already-cloned target line into the audio VAE latent and
put that latent directly into the AV denoising stream.

The speech region gets a configurable denoise mask:
  * voice_preservation = 1.0 -> mask 0.0 over speech -> speech latent is frozen exactly.
  * voice_preservation = 0.9 -> mask 0.1 -> small amount of denoising can add scene acoustics.
  * outside the speech region -> mask 1.0 -> LTX remains free to generate ambience/SFX.

Wire the output into LTXVConcatAVLatent in place of LTXVEmptyLatentAudio for stage 1. Stage 2
can continue using the audio latent coming out of stage 1, so the same voice lock carries through.
"""

from __future__ import annotations

import math

import torch
import torchaudio


class LTXVVoiceSeedLatent:
    CATEGORY = "audio/ltxv"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("audio_latent", "info")
    FUNCTION = "encode"
    DESCRIPTION = (
        "Encodes cloned dialogue into LTX's target audio latent and freezes/partially freezes "
        "only the speech region. This preserves the actual speaker voice while leaving the "
        "rest of the timeline available for native LTX ambience and SFX generation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "frames_number": (
                    "INT",
                    {"default": 121, "min": 1, "max": 10000, "step": 1},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01},
                ),
                "speech_start_seconds": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 120.0,
                        "step": 0.01,
                        "tooltip": "Where the cloned line begins in the clip.",
                    },
                ),
                "voice_preservation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "1.0 freezes the cloned voice exactly. Try 0.90-0.95 after identity "
                            "is confirmed to let LTX add more room character around speech."
                        ),
                    },
                ),
                "edge_feather_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Optional soft transition between frozen speech and generated ambience.",
                    },
                ),
            }
        }

    @staticmethod
    def _vae_sample_rate(audio_vae) -> int:
        direct = getattr(audio_vae, "audio_sample_rate", None)
        if direct:
            return int(direct)
        first_stage = getattr(audio_vae, "first_stage_model", None)
        for name in ("output_sample_rate", "sample_rate"):
            value = getattr(first_stage, name, None) if first_stage is not None else None
            if value:
                return int(value)
        return 44100

    def encode(
        self,
        audio,
        audio_vae,
        frames_number: int,
        frame_rate: float,
        speech_start_seconds: float,
        voice_preservation: float,
        edge_feather_seconds: float,
    ):
        waveform = audio["waveform"].float()
        source_rate = int(audio["sample_rate"])
        vae_rate = self._vae_sample_rate(audio_vae)
        if source_rate != vae_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, vae_rate)

        # LTX frames live on t = 0 ... (N-1)/fps, so this matches the same duration used by
        # the video timeline and by num_of_latents_from_frames().
        duration_seconds = max((int(frames_number) - 1) / float(frame_rate), 1.0 / float(frame_rate))
        target_samples = max(1, int(round(duration_seconds * vae_rate)))
        start_sample = int(round(float(speech_start_seconds) * vae_rate))
        start_sample = min(max(start_sample, 0), target_samples - 1)

        available = target_samples - start_sample
        speech_samples = min(int(waveform.shape[-1]), available)
        if speech_samples <= 0:
            raise ValueError("[LTXVVoiceSeedLatent] No room remains in the clip for the cloned speech.")
        if speech_samples < waveform.shape[-1]:
            print(
                f"[LTXVVoiceSeedLatent] WARNING: cloned line is longer than the available clip; "
                f"cropping from {waveform.shape[-1] / vae_rate:.2f}s to {speech_samples / vae_rate:.2f}s."
            )

        full = torch.zeros(
            (*waveform.shape[:-1], target_samples),
            dtype=waveform.dtype,
            device=waveform.device,
        )
        full[..., start_sample : start_sample + speech_samples] = waveform[..., :speech_samples]

        # Match ComfyUI's own LTXVReferenceAudio/LTXVAudioVAEEncode convention.
        latents = audio_vae.encode(full.movedim(1, -1))
        if latents.ndim != 4:
            raise RuntimeError(f"[LTXVVoiceSeedLatent] Expected 4D audio latent, got {tuple(latents.shape)}")

        # Force temporal latent count to exactly what the official empty-audio node would have
        # produced for this video. That prevents small waveform-duration rounding differences.
        expected_t = int(
            audio_vae.first_stage_model.num_of_latents_from_frames(int(frames_number), float(frame_rate))
        )
        current_t = int(latents.shape[2])
        if current_t > expected_t:
            latents = latents[:, :, :expected_t, :]
        elif current_t < expected_t:
            pad = torch.zeros(
                (latents.shape[0], latents.shape[1], expected_t - current_t, latents.shape[3]),
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, pad], dim=2)

        b, _c, t, _f = latents.shape
        mask = torch.ones((b, 1, t, 1), dtype=torch.float32, device=latents.device)

        speech_start = start_sample / vae_rate
        speech_end = (start_sample + speech_samples) / vae_rate
        start_idx = max(0, min(t - 1, int(math.floor((speech_start / duration_seconds) * t))))
        end_idx = max(start_idx + 1, min(t, int(math.ceil((speech_end / duration_seconds) * t))))

        speech_mask_value = float(1.0 - voice_preservation)
        mask[:, :, start_idx:end_idx, :] = speech_mask_value

        # Optional feather outside the core speech region. 1.0 is fully generated; the speech
        # value is the requested preservation amount.
        feather_latents = int(round((float(edge_feather_seconds) / duration_seconds) * t))
        if feather_latents > 0:
            for i in range(1, feather_latents + 1):
                frac = i / (feather_latents + 1)
                value = 1.0 - (1.0 - speech_mask_value) * frac
                left = start_idx - i
                right = end_idx - 1 + i
                if left >= 0:
                    mask[:, :, left, :] = torch.minimum(
                        mask[:, :, left, :], torch.tensor(value, device=mask.device, dtype=mask.dtype)
                    )
                if right < t:
                    mask[:, :, right, :] = torch.minimum(
                        mask[:, :, right, :], torch.tensor(value, device=mask.device, dtype=mask.dtype)
                    )

        info = (
            f"speech={speech_start:.2f}-{speech_end:.2f}s; preservation={voice_preservation:.2f}; "
            f"latent_frames={t}; speech_latents={start_idx}:{end_idx}; vae_sr={vae_rate}"
        )
        print(f"[LTXVVoiceSeedLatent] {info}")

        return ({"samples": latents, "noise_mask": mask, "type": "audio"}, info)


NODE_CLASS_MAPPINGS = {
    "LTXVVoiceSeedLatent": LTXVVoiceSeedLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVVoiceSeedLatent": "LTXV Voice Seed Latent",
}
