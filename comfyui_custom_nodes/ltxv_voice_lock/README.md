# LTXV Lock Character Voice (ComfyUI node)

Runs the same idea as `voice_lock/` at the repo root, but as an actual node inside your ComfyUI
graph instead of a separate terminal command: wire it in right after wherever your workflow
produces a `VIDEO` (e.g. the LTX-2.5 subgraph's output) and before `SaveVideo`, and it re-voices
just the segments where each enrolled character is speaking (up to 4 characters in one node),
leaving everyone else's dialogue untouched. Face detection/tracking runs once per video no matter
how many characters are enrolled.

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
| `character_photo` | IMAGE | Character 1's reference face photo (from a `Load Image` node). Required. |
| `character_voice` | AUDIO | Character 1's ~5-10s reference voice clip (from a `LoadAudio` node). Required. |
| `character_photo_2` / `character_voice_2` (and `_3`, `_4`) | IMAGE / AUDIO | Optional additional characters — up to 4 total in one node. Leave unconnected if you only need one. |
| `light_asd_repo` / `light_asd_weight` | STRING | Paths from step 3/4 above. |
| `converter_config` / `converter_ckpt` | STRING | Paths from step 4 above. |
| `match_threshold` | FLOAT | Min face-identity similarity to accept a match (default 0.35). Applies to every character. |
| `speaking_threshold` | FLOAT | Min Light-ASD score to count as "speaking" (default 0.0). Applies to every character. |
| `tau` | FLOAT | OpenVoice conversion strength (default 0.3). Applies to every character. |

Each enrolled character is matched to its own best-scoring face track (a track already claimed by
an earlier character can't also be claimed by a later one, so two people can't accidentally get
merged into the same re-voiced segments).

Output: `video` (VIDEO) — same clip, with each matched character's speaking segments re-voiced to
their own reference voice. Wire this into `SaveVideo` in place of your original video output.

## Honesty about accuracy and testing

Same caveats as the standalone tool (see `voice_lock/README.md`): this chains three independent
ML models and errors compound across stages. The single-character path has been run successfully
end-to-end inside a real ComfyUI instance (RTX 5090 pod) after working through a series of
environment-specific issues (missing Python packages in ComfyUI's own venv, a dead upstream
OpenVoice checkpoint URL requiring a Hugging Face mirror, a couple of old-numpy-API breakages in
Light-ASD's 2021-era code) — none of those were bugs in this node itself. The **multi-character**
path (this file's most recent change) has not yet been run end-to-end; treat your first
multi-character run as a calibration pass the same way the original single-character docs
recommended. Check the console output (it prints each character's identity-match score and
speaking segments) before trusting the result.

If a track never matches or the segments look wrong, adjust `match_threshold` /
`speaking_threshold` and re-run — recomputing a character's face embedding is cheap, so there's no
separate "enroll" step to redo. If it seems like only part of a character's dialogue got
re-voiced, that usually means Light-ASD lost continuous face tracking partway through (e.g. a
camera angle change) and split them into a second, lower-scoring track that didn't clear
`match_threshold` — try lowering `match_threshold` slightly and check the console for a second
track with a similar-but-lower similarity score.
