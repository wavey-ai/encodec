# Deterministic cross-platform LM entropy coding, acv=4 CRC chunk framing, and `_counts_from_pdf` bug fix

## Summary

This PR hardens the LM-backed entropy coding path for cross-platform correctness and adds per-segment failure isolation. The neural network weights and audio quality are unchanged. All existing `.ecdc` files decode correctly.

## Motivation

Three problems with the current LM entropy path:

1. **Non-deterministic across hardware.** `torch.softmax` can differ by a ULP between CPU, MPS, and CUDA. The arithmetic coder amplifies these differences — a single wrong probability pushes the decode state off track, producing `EOFError` or silent garbage. Payloads encoded on an Apple Silicon Mac reliably fail to decode on Linux CPU or CUDA.

2. **Silent corrupt decode at `tau=1.0`.** In `_counts_from_pdf`, the near-integer perturbation uses an alternating sign. When a token's probability is exactly `0.0` (common at `tau=1.0` due to float underflow of `exp(-large)`), the negative perturbation gives `x = -ε`, then `floor(-ε) = -1`. A negative count makes the CDF non-monotonic; the decoder produces wrong symbols with no error raised.

3. **No failure isolation.** A single corrupt byte anywhere in the payload desynchronises the arithmetic decoder and destroys the rest of the file.

## Changes

### `encodec/compress.py`

**Deterministic CDF construction**

- `_stable_softmax`: computes softmax in float64 using a sequential cumsum denominator rather than `torch.softmax`. Cross-architecture bit-reproducibility verified Mac CPU/MPS → Linux CPU/CUDA.
- `_quantize_logits_`: rounds logits to a 1/128 grid before softmax. Tiny floating-point differences that don't change the quantised logit produce identical CDFs.
- `_counts_from_pdf`: adds `clamp_min(0)` after the near-integer perturbation step, fixing the negative-count bug at `tau=1.0`.
- `_deterministic_cdf` / `_deterministic_cdf_multi`: integer floor + priority allocation CDF construction at `FP_SCALE=65536` precision. Replaces float-based CDF that was sensitive to platform differences.

**Bitstream version `acv=4` with CRC chunk framing**

- Each model segment is wrapped in `[chunk_len: u32 BE][crc32: u32 BE][payload]`.
- A corrupt chunk is replaced with silence for that segment; the rest of the file decodes normally.
- `tau` is stored in the header so encoder and decoder are always in sync without out-of-band configuration.

**GPU reliability**

- `compress_to_file` detects the model device and moves the waveform there automatically (`wav[None].to(model_device)`). Previously crashed when the model was on MPS or CUDA.
- LM and arithmetic coder always run on CPU for cross-platform determinism regardless of model device.

**Tunable defaults** (via env vars; existing behaviour unchanged if not set):

| Variable | Default |
|---|---|
| `ENCODEC_LM_TAU` | `1.0` |
| `ENCODEC_LOGIT_QSTEP` | `1/128` |
| `ENCODEC_AC_FP_SCALE` | `65536` |
| `ENCODEC_AC_MIN_RANGE` | `1` |
| `ENCODEC_DETERMINISTIC_LM_DTYPE` | `float32` |

### `encodec/model.py`

- `LMModel.forward_logits`: factored out from `forward` so the deterministic and legacy paths share the transformer forward pass.
- `LMModel.forward_legacy`: raw softmax with no quantisation, used for decoding `acv < 3` streams.
- `LMModel.__init__`: accepts `tau` parameter.
- `EncodecModel.get_lm_model`: accepts `device` and `dtype` parameters for explicit LM placement.

### `scripts/`

- `precision_eval.py`: CLI for benchmarking bitrate, SNR, encode/decode wall time, CPU vs MPS, LM vs non-LM, and single-byte corruption behaviour (targets chunk bodies, not headers/CRC).
- `payload_decode_matrix.py`: decodes a payload across CPU and CUDA and compares results; intended for cross-host determinism validation.

## Backwards compatibility

**Reading old streams: fully preserved.** The decoder reads the `acv` field from the stream header and routes accordingly:

| `acv` | Path | Notes |
|---|---|---|
| `0` | Raw bitpacking, no LM | Unchanged |
| `1` / `2` | Legacy LM via `forward_legacy()` | Original `torch.softmax`, no quantisation — decodes exactly as before |
| `4` | New deterministic path | This PR |

**Writing:** `compress(..., use_lm=False)` still produces `acv=0` raw streams identical to before. `compress(..., use_lm=True)` now produces `acv=4`; old decoders will reject `acv=4` streams with an unsupported-version error (the version field exists for this purpose).

**API surface:** no breaking changes. `compress`, `decompress`, `compress_to_file`, `decompress_from_file` retain the same signatures. The `EncodecModel` public API is unchanged.

## Test results

Benchmarked on 7 stereo 48 kHz music tracks, 10 s clips, `encodec_48khz`, all 7 tracks decoded without error on every device:

| Bandwidth | Device | Avg actual kbps | LM gain vs raw | Encode RTF | Decode RTF |
|---|---|---|---|---|---|
| 6 kbps | CPU | 4.34 | 27.7% | 0.26× | 0.27× |
| 6 kbps | MPS | 4.34 | 27.7% | 0.33× | 0.27× |
| 24 kbps | CPU | 19.3 | 19.9% | 0.39× | 0.41× |
| 24 kbps | MPS | 19.3 | 19.9% | 0.47× | 0.40× |

CPU and MPS produce byte-identical payloads and identical decoded audio (same kbps, same SNR). Zero decode failures across all tracks, bandwidths, and devices.

Cross-device decode matrix (payloads encoded on Apple Silicon Mac):

| Encode | Decode | Before | After |
|---|---|---|---|
| Mac CPU | Linux CPU | `EOFError` | ✓ |
| Mac CPU | Linux CUDA | `EOFError` | ✓ |
| Mac MPS | Linux CPU | `EOFError` | ✓ |
| Mac MPS | Linux CUDA | `EOFError` | ✓ |
