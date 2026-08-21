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
   For **either `replace` mode** (see "Node inputs" below), you also need two more packages:
   ```bash
   pip install demucs faster-whisper
   ```
   Two things worth knowing before you run these:
   - `demucs` depends on `torch>=2.1` (a floor, not a pin) — on an environment that already has a
     working `torch` (as ComfyUI's does), this should not touch your existing torch install. Run
     `python -c "import torch; print(torch.__version__)"` before and after to confirm it didn't
     change.
   - `faster-whisper` depends on `onnxruntime`, which can overlap with the `onnxruntime-gpu`
     you already installed above (they provide the same importable module). If pip swaps you from
     `onnxruntime-gpu` to plain `onnxruntime` (CPU-only), reinstall the GPU build afterward:
     `pip install --force-reinstall onnxruntime-gpu`.
   - Demucs downloads its pretrained model weights on first use from
     `dl.fbaipublicfiles.com`. This hasn't been verified reachable from every environment — if
     the first `replace`-mode run hangs or fails on a download, that's the thing to check.

   For **`replace_fishaudio` mode** (the default — recommended over `replace`, see below), you
   also need `fish-speech`. **Do not plain `pip install` it** — its `pyproject.toml` hard-pins
   `torch==2.8.0` (an exact version, not a floor like Demucs), which can silently downgrade the
   working torch your ComfyUI install already has and break it. Install with `--no-deps`, then add
   only what its actual inference code path imports (traced from source, not its full
   gradio/wandb-UI/training/API-server dependency list):
   ```bash
   pip install --no-deps fish-speech@git+https://github.com/fishaudio/fish-speech.git
   pip install hydra-core omegaconf pytorch-lightning lightning wandb loguru loralib natsort \
       pyrootutils rich transformers safetensors einops descript-audio-codec tqdm \
       typing_extensions soundfile
   python -c "import torch; print(torch.__version__)"  # confirm still your original torch version
   ```
   Two more things worth knowing:
   - `descript-audio-codec` pulls in `descript-audiotools`, which pins `protobuf<3.20` (quite
     old) — this can conflict with other packages in ComfyUI's environment that want a newer
     protobuf (e.g. `onnxruntime`, `tensorboard`). If something protobuf-related breaks after this
     install, that's the likely cause; `pip check` will show the conflict.
   - `wandb`/`pytorch-lightning` are pulled in because fish-speech's utility modules import them
     at module load time even though nothing in the actual cloning path uses them — real dead
     weight, not needed for what we're doing, but required just to make the import succeed.
3. Clone Light-ASD alongside this node (matches this node's default path widgets):
   ```bash
   cd ComfyUI/custom_nodes/ltxv_voice_lock
   git clone https://github.com/Junhua-Liao/Light-ASD.git third_party/Light-ASD
   ```
4. For **`replace` or `blend` mode**, download the OpenVoice converter checkpoint (see
   [OpenVoice's usage docs](https://github.com/myshell-ai/OpenVoice#usage)) into
   `ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/converter/` (`config.json` +
   `checkpoint.pth`). Not needed for `replace_fishaudio` mode, which doesn't use OpenVoice at all.
5. For **`replace` mode** only, also download OpenVoice's base speaker (the "neutral" voice that
   gets transcribed-text re-synthesized before conversion to your target voice) into
   `ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/base_speakers/EN/`:
   ```python
   from huggingface_hub import hf_hub_download
   import shutil
   for f in ["config.json", "checkpoint.pth", "en_default_se.pth"]:
       p = hf_hub_download("myshell-ai/OpenVoice", f"checkpoints/base_speakers/EN/{f}")
       shutil.copy(p, f"ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/base_speakers/EN/{f}")
   ```
   (Same source repo as the converter checkpoint in step 4 — just a different subfolder.) Skip
   this if you're only using `replace_fishaudio` or `blend` mode.
6. For **`replace_fishaudio` mode** (the default), download Fish Audio's `s2-pro` checkpoint into
   `ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/fish_s2pro/`:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       "fishaudio/s2-pro",
       local_dir="ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/fish_s2pro",
   )
   ```
   This is a separate model from OpenVoice's converter (step 4) — a few GB, and licensed under the
   **Fish Audio Research License**: free for research/non-commercial use, commercial use needs a
   separate license from Fish Audio directly. Skip this if you're only using `replace` or `blend`
   mode.
7. Restart ComfyUI (or use its "reload custom nodes" option). Search for **"LTXV Lock Character
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
| `light_asd_repo` / `light_asd_weight` | STRING | Paths from step 3 above. |
| `converter_config` / `converter_ckpt` | STRING | Paths from step 4 above. Only used in `replace` and `blend` modes. |
| `mode` | COMBO | `replace_fishaudio` (default), `replace`, or `blend`. See below. |
| `whisper_model` | STRING | faster-whisper model name/size used to transcribe each speaking segment in either replace mode (default `base.en`). Ignored in `blend` mode. |
| `base_speaker_config` / `base_speaker_ckpt` / `base_speaker_se` | STRING | Paths from step 5 above. Only used in `replace` mode. |
| `fish_llama_checkpoint` / `fish_decoder_checkpoint` / `fish_decoder_config_name` | STRING | Paths from step 6 above. Only used in `replace_fishaudio` mode. |
| `match_threshold` | FLOAT | Min face-identity similarity to accept a match (default 0.35). Applies to every character. |
| `speaking_threshold` | FLOAT | Min Light-ASD score to count as "speaking" (default 0.0). Applies to every character. |
| `tau` | FLOAT | OpenVoice conversion strength (default 0.3). Only used in `replace` and `blend` modes -- Fish Audio's clone has no equivalent knob. |

Each enrolled character is matched to its own best-scoring face track (a track already claimed by
an earlier character can't also be claimed by a later one, so two people can't accidentally get
merged into the same re-voiced segments).

### `replace_fishaudio` vs `replace` vs `blend` mode

- **`replace_fishaudio`** (default, recommended): for each speaking segment, separates the
  original voice from background sound (Demucs), transcribes what was said (faster-whisper), and
  re-synthesizes that text from scratch directly in your reference voice using
  [Fish Audio](https://github.com/fishaudio/fish-speech)'s zero-shot voice cloning (reference clip
  + its own transcript, one pass -- no separate tone-color conversion step), then time-fits the
  result and reinserts it over the preserved background. Fish Audio's own inference automatically
  transcribes each character's reference voice clip once at enrollment time -- no extra input
  needed from you.
- **`replace`**: the same idea, but using OpenVoice's `BaseSpeakerTTS` (generate neutral speech)
  + tone-color converter (shift its timbre toward the target voice) instead of Fish Audio. Kept as
  a fallback if you can't install `fish-speech` or don't want the extra dependency weight --
  in practice this tends to sound more robotic, since it's two lossy stages instead of one direct
  clone.
- **`blend`**: the original approach — OpenVoice's tone-color conversion applied directly to the
  original audio, with no transcription or re-synthesis at all. It keeps more of the source
  recording's exact delivery/timing/background, but because tone-color conversion only shifts
  timbre (not the underlying vocal-tract content), the result can still sound like a mix of both
  voices. Use this if you don't want the Demucs/faster-whisper/cloning dependencies at all, or if
  either replace mode's re-synthesized speech sounds too different in cadence from the original
  performance.

Both replace modes are inherently slower and more failure-prone per segment than `blend` (they
chain several models instead of one), and depend on the transcription being accurate — if
faster-whisper mishears a line, the resynthesized speech will say the wrong thing. Check the
console output, which prints the transcribed text for every segment it replaces.

Output: `video` (VIDEO) — same clip, with each matched character's speaking segments re-voiced to
their own reference voice. Wire this into `SaveVideo` in place of your original video output.

## Honesty about accuracy and testing

Same caveats as the standalone tool (see `voice_lock/README.md`): this chains multiple independent
ML models and errors compound across stages. The single-character **`blend`**-mode path has been
run successfully end-to-end inside a real ComfyUI instance (RTX 5090 pod) after working through a
series of environment-specific issues (missing Python packages in ComfyUI's own venv, a dead
upstream OpenVoice checkpoint URL requiring a Hugging Face mirror, a couple of old-numpy-API
breakages in Light-ASD's 2021-era code) — none of those were bugs in this node itself. The
**multi-character** path has been run successfully with 2 enrolled characters in `blend` mode.

**`replace` mode has been run successfully end-to-end** (single character), after fixing two real
bugs found on that first run: faster-whisper's CUDA backend needing CUDA-12-specific libraries not
present on newer CUDA stacks (now forced to CPU), and segment end-times occasionally running a few
samples past the actual audio buffer's length (now clamped). Its OpenVoice-based synthesis was
reported as sounding noticeably robotic, which led directly to adding `replace_fishaudio`.

**`replace_fishaudio` mode has not yet been run end-to-end** — it's the newest addition, and swaps
in a completely different, heavier engine (Fish Audio's `fish-speech`) for the voice-generation
step while reusing the same Demucs/faster-whisper/time-fit/remix pipeline already proven out under
`replace` mode. Treat your first `replace_fishaudio` run as a calibration pass: expect to hit real
environment issues (the `--no-deps` install's dependency list, the `protobuf` version conflict
noted in the install steps, checkpoint paths) the same way every other stage of this project did on
first run, and check the console output — it prints the transcribed text and timing for every
segment it replaces — before trusting the result.

If a track never matches or the segments look wrong, adjust `match_threshold` /
`speaking_threshold` and re-run — recomputing a character's face embedding is cheap, so there's no
separate "enroll" step to redo. If it seems like only part of a character's dialogue got
re-voiced, that usually means Light-ASD lost continuous face tracking partway through (e.g. a
camera angle change) and split them into a second, lower-scoring track that didn't clear
`match_threshold` — try lowering `match_threshold` slightly and check the console for a second
track with a similar-but-lower similarity score.
