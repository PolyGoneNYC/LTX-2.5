#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/runpod-slim
REPO=$ROOT/LTX-2.5-voice-poc
COMFY=$ROOT/ComfyUI
FACE=$COMFY/input/face.jpg
VOICE=$COMFY/input/reference_voice.wav
DATA=$ROOT/voice-adapter-data
REG=$ROOT/character_profiles
ADAPTER=$ROOT/speaker_adapter.safetensors

export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

[ -f "$FACE" ] || { echo "MISSING: $FACE"; exit 2; }
[ -f "$VOICE" ] || { echo "MISSING: $VOICE"; exit 2; }

# Direct enrollment: final user contract is face.jpg + reference_voice.wav.
uv run python packages/ltx-trainer/scripts/character_voice_router.py enroll character1 \
  --face "$FACE" --voice "$VOICE" --registry "$REG" --device cuda

# OmniVoice is only used to create varied TRAINING examples. Final film audio is
# still generated natively inside LTX, not dubbed by OmniVoice.
"$COMFY/.venv-cu128/bin/python" packages/ltx-trainer/scripts/bootstrap_voice_dataset.py \
  "$VOICE" "$DATA" --count 12

# Official LTX audio-only preprocessing + WavLM embeddings + 24GB smoke config.
uv run python packages/ltx-trainer/scripts/prepare_speaker_adapter_poc.py \
  "$DATA/dataset.json" --steps 50 --overwrite

# Train only the tiny speaker projection; the LTX base stays frozen.
uv run python packages/ltx-trainer/scripts/train_speaker_adapter.py \
  "$DATA/speaker_adapter_24gb.yaml" --speaker-mask-mode global

LATEST=$(find "$DATA/speaker_adapter_output" -type f -name 'speaker_adapter_*.safetensors' | sort | tail -1)
[ -n "${LATEST:-}" ] || { echo "NO ADAPTER CHECKPOINT FOUND"; exit 3; }
cp -f "$LATEST" "$ADAPTER"
cp -f comfyui_custom_nodes/ltxv_face_voice_adapter.py "$COMFY/custom_nodes/ltxv_face_voice_adapter.py"

echo
echo "=========================================="
echo "VOICE POC FINISHED"
echo "Character profile: $REG/character1.pt"
echo "Speaker adapter:   $ADAPTER"
echo "Comfy node:        $COMFY/custom_nodes/ltxv_face_voice_adapter.py"
echo "Restart ComfyUI, then add: LTXV Face → Character Voice Adapter"
echo "=========================================="
