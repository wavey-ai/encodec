# EnCodec: High Fidelity Neural Audio Compression

![linter badge](https://github.com/facebookresearch/encodec/workflows/linter/badge.svg)
![tests badge](https://github.com/facebookresearch/encodec/workflows/tests/badge.svg)

## Index

- [wavey-ai fork README](#wavey-ai-fork-readme)
- [Upstream README](#upstream-readme)

## wavey-ai fork README

Bottom line for the `wavey-ai` fork: on an RTX 4000 Ada, the deterministic LM path cut 48 kHz GPU encode from `99s` to `13s` on a full song, made `cuda -> cpu` decode work reliably, and slightly improved GPU decode. The trade-off is that CPU-only decode was about `48%` slower than upstream on the tested full-song run (`160.96s` vs `108.91s`).

### Precision and Robustness Improvements

This fork extends the original EnCodec with a fully deterministic, cross-platform entropy coding path plus optional native entropy-coder acceleration. The neural network weights remain unchanged.

### Bitstream version `acv=4`

When `use_lm=True`, the encoder writes bitstream version 4. Each model segment (≈1 second for the 48 kHz model) is wrapped in an independent CRC-protected chunk:

```
[chunk_len: u32 BE][crc32: u32 BE][chunk payload]
```

A single corrupt byte damages at most one chunk. The decoder substitutes silence for any chunk that fails its CRC check and continues decoding the rest of the file. Previous versions would abort the entire decode on the first error.

### Deterministic LM path

The original LM entropy path was not deterministic across hardware (MPS, CUDA, CPU), causing cross-device decode failures. The deterministic path fixes this by:

- Running the arithmetic coder on CPU and keeping the encode-side LM on CPU by default. On CUDA decode, `ENCODEC_DECODE_LM_DEVICE=auto` can run the deterministic decode LM on the model device while preserving payload compatibility.
- Computing softmax in **float64** via a sequential cumsum denominator (`_stable_softmax`) rather than platform-native `torch.softmax`, which can differ by a ULP across devices.
- **Quantising logits** to a 1/128 grid before softmax. Small floating-point differences that do not change the quantised logit produce identical CDFs.
- Building the CDF from **integer floor counts** (`FP_SCALE = 65536`) with deterministic priority allocation for the residual.
- Storing `tau` in the bitstream header so encoder and decoder are always in sync.

Cross-device decode matrix (payloads encoded on Apple Silicon Mac):

| Encode device | Decode device | Legacy (original) | This fork |
|---|---|---|---|
| Mac CPU | Linux CPU | EOFError | ✓ |
| Mac CPU | Linux CUDA | EOFError | ✓ |
| Mac MPS | Linux CPU | EOFError | ✓ |
| Mac MPS | Linux CUDA | EOFError | ✓ |

### RTX 4000 Ada results

Benchmarked on April 3, 2026 on a Linode `g2-gpu-rtx4000a1-s` instance (1x RTX 4000 Ada, 4 vCPU, Ubuntu 24.04) using `02 - Lori Asha - Westside` from the Lori Asha album premix, resampled to 48 kHz stereo, with `encodec_48khz`, `6 kbps`, and `use_lm=True`.

| Repo / case | Encode | Encode x realtime | Decode | Decode x realtime | Result |
|---|---:|---:|---:|---:|---|
| Upstream `cuda -> cuda` | `99.07 s` | `2.10x` | `116.56 s` | `1.79x` | baseline |
| Upstream `cuda -> cpu` | `98.73 s` | `2.11x` | fail | — | `RuntimeError('Binary search failed')` |
| Upstream `cpu -> cpu` | `103.81 s` | `2.01x` | `108.91 s` | `1.91x` | baseline |
| Fork `cuda -> cuda` | `13.09 s` | `15.93x` | `109.49 s` | `1.90x` | encode `7.57x` faster than upstream GPU, decode `1.06x` faster |
| Fork `cuda -> cpu` | `12.94 s` | `16.11x` | `167.56 s` | `1.24x` | cross-architecture decode succeeds |
| Fork `cpu -> cpu` | `35.22 s` | `5.92x` | `160.96 s` | `1.30x` | encode `2.95x` faster than upstream CPU, CPU decode slower |

What this means in practice:

- The biggest RTX win is encode throughput. On this full-length track, the fork cut GPU encode time from `99.07 s` to `13.09 s`.
- GPU decode is modestly faster than upstream on the same Ada card, but the main portability win is that `cuda -> cpu` decode works at all.
- CPU-only decode remains a trade-off: on the tested full-song run it was about `48%` slower than upstream (`160.96s` vs `108.91s`), but it preserves compatibility across CPU, CUDA, and Apple Silicon payload handoffs.

### Critical bug fix: `_counts_from_pdf`

At `tau=1.0`, many softmax outputs are exactly `0.0` (float underflow of `exp(-large)`). These triggered a near-integer perturbation with an alternating sign. A negative sign on `x=0.0` gives `x = -ε`, and `floor(-ε) = -1`. A negative count makes the CDF non-monotonic, causing the arithmetic decoder to produce wrong symbols silently.

Fix (one line):

```python
# Before (broken at tau=1.0):
fx = torch.floor(x)

# After (fixed):
fx = torch.floor(x.clamp_min(0))
```

This bug was present in both the original Facebook implementation and earlier revisions of this fork.

### GPU reliability

The model encoder/decoder can run on any device (CPU, MPS, CUDA). `compress_to_file` detects the model's device automatically:

```python
model_device = next(model.parameters()).device
frames = model.encode(wav[None].to(model_device))
```

### Legacy decode support

Streams from the original Facebook implementation (`acv < 3`) decode correctly via `LMModel.forward_legacy()`, which uses raw softmax with no quantisation. The decoder selects the legacy or deterministic path based on the `acv` field in the stream header.

### Tuned defaults

All settings are overridable via environment variables:

| Variable | Default | Notes |
|---|---|---|
| `ENCODEC_LM_TAU` | `1.0` | Softmax temperature. `1.0` is optimal for compression. |
| `ENCODEC_LOGIT_QSTEP` | `1/64` | Logit quantisation grid size. Slightly coarser is safer cross-host. |
| `ENCODEC_AC_FP_SCALE` | `8192` | Integer scale for CDF allocation (`2^13`). |
| `ENCODEC_AC_MIN_RANGE` | `2` | Minimum CDF range per symbol. Wider bins improve portability. |
| `ENCODEC_DETERMINISTIC_LM_DTYPE` | `float64` | LM weight dtype. `float64` is safer for cross-host determinism; `float32` is faster. |
| `ENCODEC_USE_NEAR_UNIFORM` | `0` | Enable near-uniform prior (off by default). |

### Compression results

Benchmarked on 7 stereo 48 kHz music tracks (10 s clips), `encodec_48khz`:

| Bandwidth | Device | Avg actual kbps | LM gain | Encode RTF | Decode RTF |
|---|---|---|---|---|---|
| 6 kbps | CPU | 4.34 | 27.7% | 0.26× | 0.27× |
| 6 kbps | MPS | 4.34 | 27.7% | 0.33× | 0.27× |
| 24 kbps | CPU | 19.3 | 19.9% | 0.39× | 0.41× |
| 24 kbps | MPS | 19.3 | 19.9% | 0.47× | 0.40× |

RTF < 1.0 means faster than real time. On Apple Silicon the LM still runs on CPU by default, so MPS primarily accelerates model encode/decode. On CUDA decode, `ENCODEC_DECODE_LM_DEVICE=auto` can move deterministic LM decode to the GPU, which is what the Ada benchmark above measures.

### Backward compatibility and native fast path

The repo remains backward-compatible by default:

- If the Rust module is not installed, the codec falls back to the Python entropy path.
- If the Torch C++ extension is not available, nothing breaks; it is off by default.
- Legacy payloads (`acv < 3`) still decode through the legacy path.
- Deterministic chunked payloads (`acv=4`) keep cross-device decode compatibility.

Local fallback setup, no extra toolchain required:

```bash
pip install -e .
```

That is enough to run the codec locally in pure Python.

Rust fast path, recommended:

```bash
pip install -e .
pip install maturin
cd native/encodec_ac
maturin develop --release
```

This installs the `encodec_native` module into the active virtualenv. The runtime will pick it up automatically when available.

Optional Torch C++ extension:

- This remains opt-in and is off by default.
- It requires a working C++ toolchain compatible with your local PyTorch install.
- Enable it with `ENCODEC_TORCH_EXT=1`; the extension is JIT-built on first use.
- In our testing, the Rust path is the main win. The Torch extension is optional, not required for the accelerated path.

Useful runtime knobs:

| Variable | Default | Meaning |
|---|---|---|
| `ENCODEC_NATIVE_AC` | `1` | Use the Rust arithmetic/CDF path when `encodec_native` is installed. |
| `ENCODEC_TORCH_EXT` | `0` | Enable the optional Torch C++ extension. |
| `ENCODEC_DECODE_LM_DEVICE` | `auto` | On CUDA decode, prefer GPU LM decode while preserving payload compatibility. |

### Chunk size tradeoffs

Per-segment chunk overhead is dominated by LM segmentation granularity, not the 8-byte header:

| Segment size | Approx bitrate (6 kbps, music, 4 s) | Max failure isolation |
|---|---|---|
| 1.0 s (default) | ~3600 bps | ≤ 1.0 s |
| 0.5 s | ~4050 bps | ≤ 0.5 s |
| 0.25 s | ~4600 bps | ≤ 0.25 s |

The default 1.0 s (matching the 48 kHz model segment) gives the best bitrate/isolation tradeoff.

---

## Upstream README

This is the code for the EnCodec neural codec presented in [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) [[abs]](https://arxiv.org/abs/2210.13438). We provide two multi-bandwidth models:

- A causal model operating at **24 kHz** on monophonic audio trained on a variety of audio data.
- A non-causal model operating at **48 kHz** on stereophonic audio trained on music-only data.

The 24 kHz model supports 1.5, 3, 6, 12, and 24 kbps. The 48 kHz model supports 3, 6, 12, and 24 kbps. A pre-trained language model is available for each, enabling entropy coding that reduces bitstream size by up to 40% without further quality loss.

<p align="center">
<img src="./architecture.png" alt="EnCodec architecture: convolutional+LSTM encoder, Residual Vector Quantization, convolutional+LSTM decoder, multiscale complex spectrogram discriminator, small transformer LM." width="800px"></p>

## Samples

Samples including baselines are on [our sample page](https://ai.honu.io/papers/encodec/samples.html). A quick demo of 48 kHz music with entropy coding is available by clicking the thumbnail (original tracks by [Lucille Crew](https://open.spotify.com/artist/5eLv7rNfrf3IjMnK311ByP?si=X_zD9ackRRGjFP5Y6Q7Zng) and [Voyageur I](https://open.spotify.com/artist/21HymveeIhDcM4KDKeNLz0?si=4zXF8VpeQpeKR9QUIuck9Q)).

<p align="center">
<a href="https://ai.honu.io/papers/encodec/final.mp4">
<img src="./thumbnail.png" alt="Thumbnail for the sample video."></a></p>

## 🤗 Transformers

EnCodec is available in Transformers. See the [Transformers EnCodec docs](https://huggingface.co/docs/transformers/main/en/model_doc/encodec), and the [24 kHz](https://huggingface.co/facebook/encodec_24khz) and [48 kHz](https://huggingface.co/facebook/encodec_48khz) checkpoints on the Hub.

```python
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[0]["audio"]["array"]
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
audio_codes = model(inputs["input_values"], inputs["padding_mask"]).audio_codes
```

## Installation

Requires Python 3.8+ and a recent PyTorch (1.11+ recommended; 2.x tested).

```bash
pip install -U encodec                                          # stable release
pip install -U git+https://git@github.com/wavey-ai/encodec     # this fork
pip install .                                                   # from local clone
```

For development:

```bash
pip install -e '.[dev]'
make tests
```

**Supported platforms:** macOS (Intel and Apple Silicon), recent mainstream Linux distributions. Windows is not officially supported.

## Usage

### CLI

```bash
# Compress
encodec [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] INPUT_FILE [OUTPUT_FILE]

# Decompress
encodec [-f] [-r] ENCODEC_FILE [OUTPUT_WAV_FILE]

# Round-trip (compress then immediately decompress)
encodec [-r] [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] INPUT_FILE OUTPUT_WAV_FILE
```

`--hq` selects the 48 kHz stereo model. `--lm` enables entropy coding (slower, ~20–35% smaller files).

### Python API

```python
import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.compress import compress, decompress
from encodec.utils import convert_audio

# Load model
model = EncodecModel.encodec_model_48khz()
model.set_target_bandwidth(6.0)

# Load audio (soundfile recommended over torchaudio for compatibility)
wav, sr = sf.read("audio.wav", always_2d=True, dtype="float32")
wav = torch.from_numpy(wav.T.copy())
wav = convert_audio(wav, sr, model.sample_rate, model.channels)

# Compress with LM entropy coding (acv=4, CRC chunk framing)
payload = compress(model, wav, use_lm=True)

# Decompress (works on any device; corrupt segments replaced with silence)
wav_out, out_sr = decompress(payload)
```

### GPU encode

```python
model = EncodecModel.encodec_model_48khz().to("mps")   # or "cuda"
model.set_target_bandwidth(6.0)
# compress() moves the waveform to the model device automatically;
# the LM and arithmetic coder always stay on CPU for determinism.
payload = compress(model, wav, use_lm=True)
```

### Extracting discrete codebook representations

```python
import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

wav, sr = sf.read("audio.wav", always_2d=True, dtype="float32")
wav = torch.from_numpy(wav.T.copy())
wav = convert_audio(wav, sr, model.sample_rate, model.channels)

with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([f[0] for f in encoded_frames], dim=-1)  # [B, n_q, T]
```

Codebook counts by bandwidth:

| Model | 1.5 kbps | 3 kbps | 6 kbps | 12 kbps | 24 kbps |
|---|---|---|---|---|---|
| 24 kHz mono | n_q=2 | n_q=4 | n_q=8 | n_q=16 | n_q=32 |
| 48 kHz stereo | — | n_q=2 | n_q=4 | n_q=8 | n_q=16 |

### Benchmarking and corruption testing

```bash
# Encode, decode, report bitrate/SNR/timing
python scripts/precision_eval.py \
    --repo-path . \
    --input audio.wav \
    --model encodec_48khz \
    --bandwidth 6.0 \
    --lm \
    --device mps

# Simulate a corrupt byte at the midpoint of the payload
python scripts/precision_eval.py \
    --repo-path . \
    --input audio.wav \
    --model encodec_48khz \
    --bandwidth 6.0 \
    --lm \
    --corrupt-byte-fraction 0.5

# Cross-host decode validation (run on a second machine)
python scripts/payload_decode_matrix.py --payload out.ecdc
```

---

## FAQ

**Out of memory on long files** — The model is applied to the full file at once. Split into segments manually or reduce clip length before encoding.

**DistributedDataParallel** — Not used here. Use `encodec.distrib.sync_buffer` and `encodec.distrib.sync_grad` instead.

**My `.ecdc` file from the original Facebook release won't decode** — It will. The decoder detects the bitstream version and routes `acv < 3` streams through the original LM path automatically.

**MPS is slower than CPU for encode** — The LM runs on CPU regardless of device (required for cross-platform determinism) and dominates encode time. MPS accelerates only the SEANet encoder/decoder, which is not the bottleneck at typical clip lengths.

## What's new

See [CHANGELOG.md](CHANGELOG.md) for the full history.

## Citation

```bibtex
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```

## License

MIT — see [LICENSE](LICENSE).
