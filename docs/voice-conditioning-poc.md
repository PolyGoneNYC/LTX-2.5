# LTX-2.5 In-Diffusion Voice Conditioning POC

## Goal

Teach LTX-2.5 a persistent speaker identity **inside its native audio/video diffusion pass** so the model can generate:

- the requested speaker's timbre,
- dialogue timing and mouth motion through the normal A/V transformer,
- room reverb and distance,
- ambience and Foley,
- music under dialogue,

without replacing the finished soundtrack afterward.

This is a research branch. It is not a claim that arbitrary speaker cloning already works; the branch now contains the first real trainable gradient path needed to test that hypothesis.

## Architecture

```text
clean reference speech (different utterance from target)
        |
        v
microsoft/wavlm-base-plus-sv
512-D normalized speaker x-vector
        |
        v
SpeakerEmbeddingConditioner
512 -> LTX audio hidden width (normally 2048)
        |
        v
speaker_bias x target-speech-token mask
        |
        v
LTX audio projected tokens
        |
        +--> audio self-attention
        +--> audio <-> video cross-attention
        +--> text cross-attention
        |
        v
native LTX audio + video denoising
        |
        v
voice + lips + room acoustics + ambience + music generated together
```

The speaker bias is inserted **after `audio_patchify_proj` and before the first transformer block**. The base checkpoint tensor shapes are unchanged.

## Why the speech mask matters

Speaker identity should describe *who is speaking*, not the entire soundtrack.

During a target clip such as:

```text
0.0s       1.0s             2.5s       5.0s
music  =====================================
speech          [ hello there ]
mask       000001111111111111000000000000000
```

only the speech tokens receive the speaker bias. Music, ambience and reverb-only regions receive a zero mask and remain base-LTX audio generation.

The first mask implementation uses faster-whisper word timestamps, mapped to the exact `num_time_steps` and `duration` saved in the corresponding LTX audio latent file. It adds a small 80 ms pad around words for attacks/breaths/transitions.

## What is trainable

Phase 1 freezes the entire LTX-2.5 transformer and trains only:

```text
SpeakerEmbeddingConditioner.projection
```

With the default LTX-2.5 audio hidden width this is roughly one million parameters (`512 x 2048`). The layer starts at exact zeros, so step 0 reproduces the base model and gradients gradually teach a speaker-dependent residual.

The adapter is saved separately as `speaker_adapter_XXXXXXXX.safetensors`; the 19B base model is never duplicated in the adapter checkpoint.

## Dataset contract

Use **multiple speakers and multiple utterances per speaker**. Every target row needs a clean same-speaker reference clip that is preferably a *different utterance* from the target.

Example JSON row:

```json
{
  "media_path": "clips/speaker_001_target_017.mp4",
  "caption": "A woman in a tiled hallway says, 'What are you looking at?' while soft classical music plays behind her.",
  "speaker_reference": "speaker_refs/speaker_001_ref_003.wav"
}
```

Do not use the target audio itself as its reference for a serious experiment. That leaks wording, prosody and room acoustics into the identity input and makes it easier for the adapter to memorize instead of learning speaker identity.

Recommended first useful dataset:

- 10-50 speakers
- 20+ target utterances per speaker if possible
- 4-10 second clean reference clips
- varied target rooms, distances, music and ambience
- same speaker paired with different reference utterances across target examples

A single-speaker dataset is useful only as an overfit/sanity test; it cannot prove arbitrary-speaker generalization.

## Precomputed layout

After the normal LTX joint audio/video preprocessing plus the two new scripts:

```text
DATASET_ROOT/.precomputed/
  latents/
  audio_latents/
  conditions/
  speaker_embeddings/
  speaker_masks/
```

All five trees must contain matching relative `.pt` paths.

## Step 1 - Normal LTX preprocessing

Use the repository's normal LTX-2.5 dataset preprocessing with audio enabled. The resulting audio latent files contain the exact latent time axis used later by the speaker masks.

See `packages/ltx-trainer/docs/dataset-preparation.md` and the existing low-VRAM training config for the normal preprocessing flow.

## Step 2 - Speaker embeddings

From `packages/ltx-trainer`:

```bash
python scripts/compute_speaker_embeddings.py \
  /path/to/dataset.json \
  --device cuda
```

Default model:

```text
microsoft/wavlm-base-plus-sv
```

Output:

```text
DATASET_ROOT/.precomputed/speaker_embeddings/
```

## Step 3 - Speech-token masks

```bash
python scripts/compute_speaker_masks.py \
  /path/to/dataset.json \
  --device cpu \
  --whisper-model base.en
```

Output:

```text
DATASET_ROOT/.precomputed/speaker_masks/
```

For a better-quality real dataset, use a larger Whisper model after the pipeline is proven.

## Step 4 - Train the frozen-base adapter

The trainer entry point is:

```text
packages/ltx-trainer/scripts/train_speaker_adapter.py
```

It deliberately reuses the stock joint A/V strategy and flow-matching loss, while overriding trainable parameter collection so only the speaker projection updates.

For a 32 GB 5090, start from `configs/t2v_lora_low_vram.yaml` and make these changes:

```yaml
model:
  training_mode: "lora"   # keeps the stock low-VRAM/quantization path; this script does NOT install LoRA

training_strategy:
  name: "flexible"
  video:
    is_generated: true
    latents_dir: "latents"
  audio:
    is_generated: true
    latents_dir: "audio_latents"

optimization:
  batch_size: 1
  learning_rate: 1.0e-4
  steps: 100   # first smoke test only

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

validation:
  interval: null   # stock validation does not inject the speaker adapter yet

checkpoints:
  interval: 25

output_dir: "outputs/speaker_adapter_poc"
```

Run:

```bash
python scripts/train_speaker_adapter.py /path/to/speaker_adapter.yaml
```

## What a successful smoke train must prove

Before spending money on a long run, confirm all of these:

1. `Speaker-adapter trainable params` is about one million, not billions.
2. The first backward pass completes without accidentally unfreezing base LTX.
3. `speaker_adapter_*.safetensors` is small and contains only `speaker_adapter.*` tensors.
4. The projection weight norm moves above zero after optimization.
5. Training loss is finite and gradients on the projection are non-zero.
6. An all-zero speech mask produces base-LTX behavior and no speaker-conditioned residual.

## Evaluation design

Do not judge only by whether one clip 'sounds similar.' Use held-out speakers and held-out sentences.

For each evaluation prompt generate with:

- speaker A embedding
- speaker B embedding
- zero/no speaker embedding

while keeping prompt, seed and initial image identical.

Measure/listen for:

- speaker verification similarity between generated dialogue and reference
- intelligibility / transcript accuracy
- lip-motion quality
- preservation of music/ambience/reverb
- whether switching only the speaker embedding changes voice identity without changing scene acoustics

The critical success test for this project is exactly the intended use case: **classical music generated by LTX continuously underneath dialogue while the requested voice identity remains stable and the room sound remains native.**

## Phase 2 if the 1M projection is too weak

A single additive projection is intentionally the smallest possible experiment. If it learns identity weakly but measurably, extend capacity in this order:

1. two-layer MLP projection with zero-initialized output layer;
2. per-block learned speaker scale/gate on the audio stream;
3. low-rank speaker-conditioned adapters on audio self-attention Q/K/V;
4. low-rank speaker-conditioned adapters on A/V cross-attention so speaker state more strongly influences facial motion;
5. auxiliary speaker-similarity loss on decoded/generated speech segments.

Do **not** jump directly to full-model fine-tuning until the small adapter has established a measurable speaker signal.

## Files added/changed on `voice-conditioning-poc`

Core:

- `packages/ltx-core/src/ltx_core/conditioning/character_voice.py`
- `packages/ltx-core/src/ltx_core/conditioning/speaker_embedding.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/tests/conditioning/test_speaker_embedding.py`
- `packages/ltx-core/tests/conditioning/test_transformer_speaker_conditioning.py`

Trainer/preprocessing:

- `packages/ltx-trainer/scripts/compute_speaker_embeddings.py`
- `packages/ltx-trainer/scripts/compute_speaker_masks.py`
- `packages/ltx-trainer/scripts/train_speaker_adapter.py`

## Current status

The branch now has a **real trainable conditioning path** from an external speaker vector into LTX audio transformer tokens. It has not yet demonstrated learned voice cloning; the next milestone is a small overfit/smoke training run, followed by inference wiring that loads the saved adapter and applies a reference embedding during normal LTX-2.5 generation.
