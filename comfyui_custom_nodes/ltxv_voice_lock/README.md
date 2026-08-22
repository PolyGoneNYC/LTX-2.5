# LTXV Lock Character Voice (ComfyUI node)

Runs the same idea as `voice_lock/` at the repo root, but as an actual node inside your ComfyUI
graph instead of a separate terminal command: wire it in right after wherever your workflow
produces a `VIDEO` (e.g. the LTX-2.5 subgraph's output) and before `SaveVideo`, and it re-voices
just the segments where each enrolled character is speaking, leaving everyone else's dialogue
untouched. Face detection/tracking runs once per video no matter how many characters are enrolled.

Characters come from either (or both) of two places: a **character library folder** on disk
(`character_folder`), so recurring characters don't have to be re-wired for every shot, and up to
4 photo+voice pairs wired directly into the node's `IMAGE`/`AUDIO` inputs.

The node outputs the re-voiced `VIDEO` **and** the new `AUDIO` on its own. That second output is
there so the new voice can be fed to a lip-sync node, which redraws each mouth to match it — see
[Lip-sync: making the mouths match](#lip-sync-making-the-mouths-match) below.

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

   For **`replace_omnivoice` mode** (the default — recommended over the other two re-synthesis
   modes, see below), you need the `omnivoice` package:
   ```bash
   pip install omnivoice
   ```
   Unlike `fish-speech` below, OmniVoice's published dependencies only require `torch>=2.4` (a
   floor, not an exact pin), so on an environment that already has a working, newer torch (as
   ComfyUI's does) a plain install should leave it alone -- still worth confirming with
   `python -c "import torch; print(torch.__version__)"` before and after. One thing to watch:
   OmniVoice requires `transformers>=5.3.0`; if ComfyUI's environment has an older `transformers`
   (e.g. pinned by another custom node), pip *will* try to upgrade it, which can cascade into
   conflicts the same way the `protobuf` version chain did for `fish-speech` below -- check
   `pip check` after installing if anything else breaks.

   For **`replace_fishaudio` mode** (kept as an alternative), you
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
6. For **`replace_omnivoice` mode** (the default), no manual download step is needed --
   `omnivoice_model_path` defaults to the HuggingFace Hub repo id `k2-fsa/OmniVoice`, and
   `OmniVoice.from_pretrained()` downloads and caches it automatically on first use (into HF's
   usual `~/.cache/huggingface` cache, not this node's `checkpoints/` folder). If you'd rather
   pre-download it (e.g. for an offline pod), point `omnivoice_model_path` at a local directory
   instead:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       "k2-fsa/OmniVoice",
       local_dir="ComfyUI/custom_nodes/ltxv_voice_lock/checkpoints/omnivoice",
   )
   ```
   Skip this if you're only using `replace_fishaudio`, `replace`, or `blend` mode.
7. For **`replace_fishaudio` mode**, download Fish Audio's `s2-pro` checkpoint into
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
8. Restart ComfyUI (or use its "reload custom nodes" option). Search for **"LTXV Lock Character
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
| `character_folder` | STRING | Path to a character library on disk (see below). Leave empty to use only the wired inputs. |
| `character_photo` | IMAGE | Optional. Character 1's reference face photo (from a `Load Image` node). |
| `character_voice` | AUDIO | Optional. Character 1's ~5-10s reference voice clip (from a `LoadAudio` node). |
| `character_photo_2` / `character_voice_2` (and `_3`, `_4`) | IMAGE / AUDIO | Optional additional wired characters — up to 4 pairs. Leave unconnected if you only need one, or none at all if you're using `character_folder`. |
| `light_asd_repo` / `light_asd_weight` | STRING | Paths from step 3 above. |
| `converter_config` / `converter_ckpt` | STRING | Paths from step 4 above. Only used in `replace` and `blend` modes. |
| `mode` | COMBO | `replace_omnivoice` (default), `replace_fishaudio`, `replace`, or `blend`. See below. |
| `whisper_model` | STRING | faster-whisper model name/size used to transcribe each speaking segment in any replace mode (default `base.en`). Ignored in `blend` mode. |
| `base_speaker_config` / `base_speaker_ckpt` / `base_speaker_se` | STRING | Paths from step 5 above. Only used in `replace` mode. |
| `fish_llama_checkpoint` / `fish_decoder_checkpoint` / `fish_decoder_config_name` | STRING | Paths from step 7 above. Only used in `replace_fishaudio` mode. |
| `omnivoice_model_path` | STRING | Local directory or HF Hub repo id from step 6 above (default `k2-fsa/OmniVoice`). Only used in `replace_omnivoice` mode. |
| `match_threshold` | FLOAT | Min face-identity similarity to accept a match (default 0.35). Applies to every character. |
| `speaking_threshold` | FLOAT | Min Light-ASD score to count as "speaking" (default 0.0). Applies to every character. |
| `min_segment_seconds` | FLOAT | Speaking segments shorter than this are ignored (default 0.4). |
| `device` | STRING | Torch device for the conversion/cloning models (default `cuda:0`). |
| `tau` | FLOAT | OpenVoice conversion strength (default 0.3). Only used in `replace` and `blend` modes -- neither OmniVoice's nor Fish Audio's clone has an equivalent knob. |

Each enrolled character is matched to its own best-scoring face track (a track already claimed by
an earlier character can't also be claimed by a later one, so two people can't accidentally get
merged into the same re-voiced segments).

### The character library (`character_folder`)

Point `character_folder` at a directory laid out with one sub-directory per character:

```
characters/
  cleopatra/
    face.jpg
    voice.wav
  guard/
    face.png
    voice.mp3
```

The sub-directory **name is the character's name** (it's what shows up in the console log). Inside
each one, the first image file (`.jpg` `.jpeg` `.png` `.webp` `.bmp`) is the reference face and the
first audio file (`.wav` `.mp3` `.flac` `.ogg` `.m4a`) is the reference voice. Order is
alphabetical, so if you keep several images in a folder, name the one you want `face.jpg` or
similar so it sorts first.

Every character in the library is enrolled on every run: the node scans them all, computes a face
embedding for each, and matches each against the face tracks it found in the video. Characters who
aren't in this particular shot simply don't match anything and are skipped with a console line
saying so — there's no cost to keeping a large library. A sub-directory missing either a face or a
voice is reported and skipped rather than failing the run.

The library is unbounded; only the *wired* `character_photo`/`character_voice` inputs are capped at
4 (that's a limit of how many input sockets the node exposes, not of the matching itself).
Characters from both sources are enrolled together.

### `replace_omnivoice` vs `replace_fishaudio` vs `replace` vs `blend` mode

- **`replace_omnivoice`** (default, recommended): for each speaking segment, separates the
  original voice from background sound (Demucs), transcribes what was said (faster-whisper), and
  re-synthesizes that text from scratch directly in your reference voice using
  [OmniVoice](https://github.com/k2-fsa/OmniVoice)'s zero-shot voice cloning (reference clip +
  its transcript, one pass). Unlike Fish Audio below, OmniVoice's `generate()` takes a `duration`
  parameter that natively targets the segment's exact length, so the result needs far less
  post-hoc time-stretching to fit -- large phase-vocoder stretches are the main source of
  robotic-sounding replaced speech, so this is the mode expected to sound most natural.
- **`replace_fishaudio`** (kept as an alternative): the same idea, using
  [Fish Audio](https://github.com/fishaudio/fish-speech)'s zero-shot voice cloning (reference clip
  + its own transcript, one pass -- no separate tone-color conversion step) instead of OmniVoice.
  Fish Audio's API has no duration/speed control at all, so its output length can land far from
  the segment's target and lean much harder on time-stretching to compensate. Fish Audio's own
  inference automatically transcribes each character's reference voice clip once at enrollment
  time -- no extra input needed from you.
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

All replace modes are inherently slower and more failure-prone per segment than `blend` (they
chain several models instead of one), and depend on the transcription being accurate — if
faster-whisper mishears a line, the resynthesized speech will say the wrong thing. Check the
console output, which prints the transcribed text for every segment it replaces.

Outputs:

- `video` (VIDEO) — same clip, with each matched character's speaking segments re-voiced to their
  own reference voice. Wire this into `SaveVideo` in place of your original video output.
- `audio` (AUDIO) — the new soundtrack on its own (re-voiced speech plus the original background).
  Feed this to a lip-sync node together with the video; see below.

## Lip-sync: making the mouths match

This node changes **what is heard**, not what the mouths do. In every replace mode the original
performance is thrown away and a new one is generated, so the new speech lands on its own cadence
— close to the original in total duration, and (since phrase-level re-dubbing) anchored per phrase
to the original word timings, but not frame-accurate against the lips. `blend` mode is the one
exception: it converts timbre on the *original waveform* without touching timing, so it stays in
sync by construction, at the cost of still sounding partly like the original speaker.

The fix is not to keep forcing the audio to fit fixed lips. It's to do it the other way round:
generate the voice you want, then **redraw the mouth to match that voice**. That's what dedicated
lip-sync models do, and it's the direction they were actually built for.

```
  ...your LTX-2.5 video  ──→  LTXV Lock Character Voice  ──┬── video ──→  ┐
                                                           │              ├─→  lip-sync node  ──→  SaveVideo
                                                           └── audio ──→  ┘
```

The lip-sync node is a separate custom node — this package doesn't bundle one. Working options,
all with ComfyUI wrappers:

| Model | Notes |
|---|---|
| [LatentSync](https://github.com/bytedance/LatentSync) (ByteDance) | Latent-space; best identity preservation of the three, so faces stay looking like themselves. |
| [MuseTalk](https://github.com/TMElyralab/MuseTalk) (Tencent) | Real-time (30+ FPS) on a 256×256 face region; fastest. |
| [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Oldest, and still the strongest raw sync (it's trained against a SyncNet discriminator), but lowest output resolution. |

Install one of their ComfyUI wrappers into `ComfyUI/custom_nodes/`, then wire this node's two
outputs into it. Which one to pick is a quality trade-off, not a correctness one: try LatentSync
first if faces matter more, Wav2Lip if sync tightness matters more.

**This half has not been run end-to-end here** — no lip-sync node is installed on the pod this was
developed against. The wiring above is the intended architecture, not a verified run.

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

**`replace_fishaudio` mode has been run successfully end-to-end** (single character) after fixing
several real environment issues along the way (the `--no-deps` install's dependency list, a
`protobuf` version conflict, a `torchcodec`/`torchaudio` version mismatch, and a cross-run VRAM
leak from Fish Audio's own `launch_thread_safe_queue()` spawning a new persistent model thread on
every node execution -- now fixed by caching the built engine). Once matching succeeded, the
resulting voice was reported as still sounding robotic: Fish Audio's API has no duration/speed
parameter, so its output length regularly landed far from the segment's target and needed a large
phase-vocoder stretch to fit, which is what actually produced the artifacts. That's what led
directly to adding `replace_omnivoice`.

**`replace_omnivoice` mode has been run successfully end-to-end** (single character). The voice
quality was reported as clearly the best of the replace modes — natural, not robotic — after three
further fixes found on real runs: OmniVoice's `generate()` defaults to inserting leading silence
(now `pad_duration=0.0`, which was the cause of the replaced line starting late), Demucs' default
`htdemucs` leaving enough vocal energy in the "background" residual to hear a ghost of the original
voice underneath the new one (now the `htdemucs_ft` ensemble), and Light-ASD's frame indices being
converted with a hardcoded 25 fps when its own ffmpeg extraction actually samples at the video's
native rate (now read from the file), which was making face matching fail outright on non-25fps
clips.

**It still does not lip-sync**, and neither do `replace` or `replace_fishaudio` — that's structural,
not a bug, and it's why the lip-sync stage above exists. Re-dubbing at phrase level (each phrase
anchored to its own faster-whisper word timestamps, rather than one TTS call per whole speaking
turn) tightened it noticeably but does not close the gap: an independently generated performance
paced by a TTS engine will not land syllable-for-syllable on mouth movements it never saw.

### Three approaches that were tried and don't work

Before settling on "generate the voice, then redraw the mouth", three ways of doing it inside LTX
itself were tried on real runs. All three failed, and they're recorded here so they don't get
re-attempted:

1. **Freeze the video latent, regenerate only the audio** (per-token `denoise_mask` = 0 on video,
   1 on audio, via `LTXVConcatAVLatent`'s per-modality nested mask). The plumbing works exactly as
   documented and the graph runs — but the model was never trained for video-conditioned audio
   inpainting, and every shipped LTX pipeline runs the opposite direction. Result: audio that
   ignores the frozen footage entirely.
2. **`LTXVReferenceAudio`** — inject a reference voice clip as clean context tokens during
   generation. The model code path is real and generic, but the speaker identity it's meant to
   carry lives in ID-LoRA weights that exist only for **LTX-2.3**. On base 2.5 the reference is
   effectively ignored; pushing `identity_guidance_scale` up to compensate produces artifacts
   rather than the target voice.
3. **The Dub-It IC-LoRA pipeline** — reads its voice reference from the *same* container as the
   reference video (so it preserves the source speaker rather than substituting a new one),
   regenerates video from pure noise, and requires an LTX-2.3-only LoRA. Not applicable.

The common blocker: **no reference-voice / voice-cloning LoRA exists for LTX-2.5 from anyone**, and
a LoRA only works with the model it was trained on. All three approaches were trying to force
*audio* to conform to fixed lips, which is the unsupported direction. The lip-sync stage reverses
it.

If a track never matches or the segments look wrong, adjust `match_threshold` /
`speaking_threshold` and re-run — recomputing a character's face embedding is cheap, so there's no
separate "enroll" step to redo. If it seems like only part of a character's dialogue got
re-voiced, that usually means Light-ASD lost continuous face tracking partway through (e.g. a
camera angle change) and split them into a second, lower-scoring track that didn't clear
`match_threshold` — try lowering `match_threshold` slightly and check the console for a second
track with a similar-but-lower similarity score.
