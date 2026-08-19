# LTXV Lock Character Voice (ComfyUI node)

Runs the same idea as `voice_lock/` at the repo root, but as an actual node inside your ComfyUI
graph instead of a separate terminal command: wire it in right after wherever your workflow
produces a `VIDEO` (e.g. the LTX-2.5 subgraph's output) and before `SaveVideo`, and it re-voices
just the segments where your enrolled character is speaking.

## Install

1. Copy this whole `ltxv_voice_lock/` folder into `ComfyUI/custom_nodes/`.
2. Install its dependencies **into ComfyUI's own Python environment** (not a separate venv this
   time — the node runs inside the ComfyUI process, so it needs to import these directly):
   ```bash
   # from ComfyUI's own venv/environment
   pip install insightface onnxruntime-gpu opencv-python librosa soundfile
   pip install git+https://github.com/myshell-ai/OpenVoice.git
   ```
3. Clone Light-ASD alongside this node (matches this node's default path widgets):
   ```bash
   cd ComfyUI/custom_nodes/ltxv_voice_lock
   git clone https://github.com/Junhua-Liao/Light-ASD.git third_party/Light-ASD
   ```
4. Download the OpenVoice converter checkpoint (see
   [OpenVoice's usage docs](https://github.com/myshell-ai/OpenVoice#usage)) into
   `ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/converter/` (`config.json` +
   `checkpoint.pth`).
5. Restart ComfyUI (or use its "reload custom nodes" option). Search for **"LTXV Lock Character
   Voice"** — it should now appear under `audio/ltxv`.

**Real risk of this path (installing into ComfyUI's own environment, versus the standalone
`voice_lock/` tool's isolated venv):** insightface, Light-ASD, and OpenVoice pin their own
torch/torchvision/onnxruntime/numpy versions, which can conflict with what ComfyUI itself needs
and, in the worst case, break your existing ComfyUI install. There's no way around this and still
have it run inside the graph — if that risk is unacceptable, use the standalone `voice_lock/`
tool instead (isolated venv, runs as a step after ComfyUI instead of inside it).

**License note:** InsightFace's pretrained recognition weights are non-commercial-use only (its
code is MIT). See `voice_lock/README.md` for the same note in more detail.

## Node inputs

| Input | Type | What it is |
|---|---|---|
| `video` | VIDEO | Your generated clip, wherever your workflow produces it. |
| `character_photo` | IMAGE | One reference photo of the character's face (from a `Load Image` node). |
| `character_voice` | AUDIO | ~5-10s reference clip of their voice (from a `LoadAudio` node). |
| `light_asd_repo` / `light_asd_weight` | STRING | Paths from step 3/4 above. |
| `converter_config` / `converter_ckpt` | STRING | Paths from step 4 above. |
| `match_threshold` | FLOAT | Min face-identity similarity to accept a match (default 0.35). |
| `speaking_threshold` | FLOAT | Min Light-ASD score to count as "speaking" (default 0.0). |
| `tau` | FLOAT | OpenVoice conversion strength (default 0.3). |

Output: `video` (VIDEO) — same clip, with the matched character's speaking segments re-voiced.
Wire this into `SaveVideo` in place of your original video output.

## Honesty about accuracy and testing

Same caveats as the standalone tool (see `voice_lock/README.md`): this chains three independent
ML models, errors compound across stages, and this specific ComfyUI-node wrapper has **not been
run inside an actual ComfyUI instance** (no GPU/ComfyUI available in the environment this was
built in) — its tensor conversions were verified against ComfyUI's real source
(`comfy_api/latest/_input_impl/video_types.py`, `comfy_extras/nodes_video.py`,
`comfy_extras/nodes_audio.py`) rather than guessed, but treat your first run as a calibration
pass. Check the console output (it prints each track's identity-match score and the speaking
segments it found) before trusting the result.

If a track never matches or the segments look wrong, adjust `match_threshold` /
`speaking_threshold` and re-run — recomputing the character's face embedding is cheap, so there's
no separate "enroll" step to redo.
