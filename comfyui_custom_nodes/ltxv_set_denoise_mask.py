"""ComfyUI custom node: freeze or free an entire LTXV latent stream (video or audio).

Install: copy this file to ``ComfyUI/custom_nodes/ltxv_set_denoise_mask.py`` and restart
ComfyUI. A new node "LTXV Set Denoise Mask" appears under category ``model/latent/ltxv``,
next to ComfyUI's own Concat/Separate AV Latent nodes.

WHY THIS EXISTS
---------------
ComfyUI already has everything needed to regenerate ONE modality of an LTX audio+video
latent while holding the other perfectly still:

* ``LTXVConcatAVLatent`` merges a video latent and an audio latent, and if either carries a
  ``noise_mask`` it builds a per-modality NestedTensor mask (comfy_extras/nodes_lt.py).
* the sampler unbinds that nested mask and applies each half to its own stream
  (comfy/samplers.py, ``if denoise_mask.is_nested``), where per-token
  ``out = out * denoise_mask + latent_image * (1 - denoise_mask)`` means **0 preserves the
  original exactly and 1 regenerates it**.
* ``LTXVSeparateAVLatent`` pulls the streams back apart afterwards.

The only missing piece is producing the mask itself. Core's ``SetLatentNoiseMask`` takes a
MASK and reshapes it to ``(-1, 1, H, W)`` -- an image-shaped, spatial mask, which is not the
right shape for a 5D video latent or an audio latent, and offers no way to say "all of it".
This node just writes a constant ``noise_mask`` shaped exactly like the latent's own samples,
which is precisely what ``LTXVConcatAVLatent`` expects (it falls back to ``torch.ones_like``).

THE GRAPH (native in-diffusion voice replacement, no post-processing)
--------------------------------------------------------------------
    LoadVideo ─→ VAEEncode (video vae) ──→ [LTXV Set Denoise Mask: 0.0] ─┐
                                                                         ├─→ LTXVConcatAVLatent
    LTXVEmptyLatentAudio ───────────────→ [LTXV Set Denoise Mask: 1.0] ─┘            │
                                                                                     ▼
                                                              SamplerCustomAdvanced (LTX AV model)
                                                                                     │
                                                                       LTXVSeparateAVLatent
                                                                                     │ (audio)
                                                                        LTXVAudioVAEDecode
                                                                                     │
                                                                            SaveAudio / mux

Video mask 0.0 pins your existing footage at sigma=0 for the whole run; audio mask 1.0 lets
the audio denoise from noise while attending to those clean video tokens every step. The
model does the lip-sync itself instead of us fitting speech to whisper timings afterwards.

Take the AUDIO from ``LTXVSeparateAVLatent`` and mux it onto your ORIGINAL video file. The
sampler's video output is only a VAE round-trip of footage you already have (it was frozen),
so it is strictly softer than your original -- there is no reason to keep it.
"""

import torch


class LTXVSetDenoiseMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "denoise_mask": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "0.0 = freeze this stream completely (output is the input latent, "
                            "bit-exact, no drift across steps). 1.0 = regenerate it completely. "
                            "Values in between blend the denoised result with the original."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"
    CATEGORY = "model/latent/ltxv"
    DESCRIPTION = (
        "Write a constant per-token denoise mask over an entire LTXV latent stream, so one "
        "modality can be held perfectly still while the other regenerates. Feed the result "
        "into LTXVConcatAVLatent, which combines the two streams' masks into the per-modality "
        "nested mask the sampler applies separately to each. Freeze video (0.0) + free audio "
        "(1.0) to regenerate only the soundtrack against fixed footage."
    )

    def set_mask(self, latent, denoise_mask):
        out = latent.copy()
        samples = out["samples"]
        # Match the samples' own shape exactly -- that is what LTXVConcatAVLatent assumes when
        # it pairs the two streams' masks (it uses torch.ones_like(samples) as its default),
        # and it avoids core SetLatentNoiseMask's image-shaped (-1, 1, H, W) reshape, which
        # does not describe a 5D video latent or an audio latent.
        out["noise_mask"] = torch.full_like(samples, float(denoise_mask))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "LTXVSetDenoiseMask": LTXVSetDenoiseMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVSetDenoiseMask": "LTXV Set Denoise Mask",
}
