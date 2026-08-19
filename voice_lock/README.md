# voice_lock

A post-processing tool that runs *after* LTX-2.5 has already generated a video: it finds where a
specific named character is the one actually talking (not just on screen), and replaces the
generated audio in those segments with a version re-voiced to match a reference voice sample —
so "Farley the hillbilly" always sounds like Farley, regardless of what voice the model happened
to assign him during generation.

## Why this exists, and why it's not a change to LTX-2.5 itself

LTX-2.5's audio conditioning (`LTXVReferenceAudio` in ComfyUI / `AudioConditionByReferenceLatent`
in this repo's `ltx-core`) only supports **one global reference voice per generation, for the
whole clip** — verified against both this repo's source and ComfyUI's actual model code
(`comfy/ldm/lightricks/av_model.py`). There is no way to scope a reference voice to a specific
on-screen character or to a specific time segment; the model has no concept of "faces" at all.
This is confirmed as a known limitation in Lightricks' own LTX-2 paper. So a 4-person scene where
only one named character should have a specific voice can't be solved by changing how the
diffusion model is conditioned — it needs to happen as a separate step after generation.

## What it does, stage by stage

1. **Enroll** (`enroll_character.py`) — given one or more reference photos of a character and a
   short (~5-10s) reference voice clip, compute and save a face-identity embedding + the voice
   clip path as that character's "profile."
2. **Find active-speaker segments** (`lock_character_voice.py`, stage 1) — runs
   [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) on the generated video. It does its own
   face detection/tracking and outputs, per detected face *track*, a per-frame "is this face
   talking right now" confidence score.
3. **Identify which track is your character** (stage 2) — for each face track, crops a
   representative frame and runs [InsightFace](https://github.com/deepinsight/insightface) to get
   its identity embedding, then compares it (cosine similarity) against the enrolled character's
   embedding. Combined with stage 1's scores, this produces a timeline of "seconds where the
   enrolled character is confirmed speaking."
4. **Re-voice those segments** (stage 3) — for each confirmed segment, uses
   [OpenVoice](https://github.com/myshell-ai/OpenVoice)'s zero-shot tone-color converter to
   convert that segment's generated speech to match the enrolled reference voice, preserving the
   original timing/prosody (so lip movements the model already generated still roughly line up).
5. **Splice** — replaces those segments in the audio track (short crossfades at the boundaries)
   and re-muxes with the original video frames into the final output.

## Install

These are three separate, fairly heavy research projects with their own dependency trees that can
conflict (especially torch/torchvision/onnxruntime pins). **Recommend a dedicated virtualenv for
this tool, separate from your LTX-2.5 environment:**

```bash
python3 -m venv .venv-voicelock
source .venv-voicelock/bin/activate
pip install insightface onnxruntime-gpu opencv-python librosa soundfile scenedetect \
    git+https://github.com/myshell-ai/OpenVoice.git \
    torch torchaudio  # match your CUDA version
git clone https://github.com/Junhua-Liao/Light-ASD.git third_party/Light-ASD
```

Download checkpoints:
- InsightFace: auto-downloads its default model pack (`buffalo_l`) on first `FaceAnalysis` use.
- Light-ASD: pretrained weight ships in their repo at `weight/pretrain_AVA_CVPR.model` (or the
  more accurate `weight/finetuning_TalkSet.model`, see their README).
- OpenVoice: download the `checkpoints/converter` config + checkpoint from their repo/HF release
  (see [OpenVoice README](https://github.com/myshell-ai/OpenVoice#usage)) — path this in
  `--converter-config` / `--converter-ckpt`.

**License note:** InsightFace's *code* is MIT, but its pretrained recognition weights are
non-commercial-use only. If this is for commercial content, either train/license a commercial
face-recognition model or confirm your use case is covered.

## Usage

```bash
# One-time per character
python enroll_character.py \
    --name farley \
    --reference-photo farley_face1.jpg --reference-photo farley_face2.jpg \
    --reference-voice farley_voice.wav \
    --profile-dir profiles/

# After every LTX-2.5 generation featuring Farley
python lock_character_voice.py \
    --input-video generated_scene.mp4 \
    --character-profile profiles/farley.json \
    --light-asd-repo third_party/Light-ASD \
    --light-asd-weight third_party/Light-ASD/weight/finetuning_TalkSet.model \
    --converter-config checkpoints/converter/config.json \
    --converter-ckpt checkpoints/converter/checkpoint.pth \
    --output-video generated_scene.farley_voice.mp4
```

## Honesty about accuracy

This chains three independent ML models, each with real error rates: face matching can misfire on
similar-looking characters or bad angles, active-speaker detection can miss fast cuts or
overlapping speech, and voice conversion can sound artificial or drift on unusual prosody. Errors
compound across stages. This was built by reading each library's real source and current API (not
guessed from memory) but has **not been run end-to-end** — there's no GPU or test video available
in the environment this was built in. Treat the first few runs as a calibration pass: check the
printed per-track identity-match scores and speaking-segment timestamps before trusting the final
output, and tune `--match-threshold` / `--speaking-threshold` (see `--help`) against what you see.
