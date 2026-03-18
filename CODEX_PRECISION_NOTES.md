# Codex Precision Notes

## Scope

Work in this branch is focused on EnCodec robustness across PyTorch versions and hardware, especially for LM-backed entropy coding, while trying to recover bitrate and runtime losses introduced by the deterministic path.

All work for this thread stays in `/Users/jamieb/wavey.ai/codex/encodec-codex`.

## Environment

- Python: `3.10.20`
- Torch: `2.10.0`
- Torchaudio: `2.10.0`
- Test corpus: `~/Downloads/Lori EP`
- Primary comparison model: `encodec_48khz`, `6 kbps`

### Cross-host Validation Environment

- Source host: Apple Silicon Mac, tested on both `cpu` and `mps`
- Decode host: Linode `g2-gpu-rtx4000a1-s`
- Region: `de-fra-2`
- GPU: `NVIDIA RTX 4000 Ada Generation`
- Remote OS: Ubuntu `24.04.3`
- Remote Python: `3.10.20`
- Remote Torch: `2.6.0+cu124`
- Remote Torchaudio: `2.6.0+cu124`

## Findings So Far

### Confirmed

- The deterministic arithmetic path fixes the main `mps -> cpu` LM decode failure seen in the legacy implementation.
- The current deterministic implementation regresses LM compression efficiency and runtime.
- Simple model segmentation is not enough for failure containment. Without explicit chunk framing, single-bit corruption can still desynchronize the decoder.
- Backward compatibility matters. The deterministic decoder initially rejected legacy streams and needed a legacy decode path.

### Baseline Measurements

Initial measurements on 3 Lori EP clips, 6 seconds each, CPU, `encodec_48khz`, `6 kbps`:

- Baseline, no LM: `6186.7 bps`, encode `0.161s`, decode `0.204s`
- Current deterministic branch, no LM: `6216.0 bps`, encode `0.171s`, decode `0.216s`
- Baseline, LM: `4148.9 bps`, encode `1.310s`, decode `1.411s`
- Current deterministic branch, LM: `4456.4 bps`, encode `1.874s`, decode `1.932s`

Interpretation:

- Non-LM path is effectively fine.
- LM path is where the regression sits.

## Branch Changes Made In This Iteration

### Robustness and Compatibility

- Added a legacy LM decode path so old `.ecdc` streams still decode.
- Split deterministic and legacy LM probability generation.
- Verified current branch can decode a legacy LM stream produced by the baseline checkout.
- Added a chunk-framed bitstream version `acv=4` with per-segment length and CRC.
- For `acv=4`, chunk decode failures now fall back to silence for that chunk instead of aborting the full decode.

### Experiment Support

- Added `scripts/precision_eval.py` to benchmark:
  - bitrate
  - encode/decode wall time
  - CPU vs MPS
  - LM vs non-LM
  - segment size effects
  - single-byte corruption behavior
- Extended `scripts/precision_eval.py` so it can:
  - write clean payloads directly
  - write the corrupted payload it actually tested
  - target chunk body bytes for `acv=4` instead of accidentally flipping chunk headers or CRC bytes
- Added `scripts/payload_decode_matrix.py` to decode payloads across `cpu` and `cuda` and compare clean/corrupt pairs on a second host
- Parameterized deterministic coder settings via env vars:
  - `ENCODEC_DETERMINISTIC_LM_DTYPE`
  - `ENCODEC_LOGIT_QSTEP`
  - `ENCODEC_LM_TAU`
  - `ENCODEC_AC_FP_SCALE`
  - `ENCODEC_AC_MIN_RANGE`
  - `ENCODEC_USE_NEAR_UNIFORM`

## Tuned Defaults Chosen

The deterministic branch now defaults to:

- `ENCODEC_DETERMINISTIC_LM_DTYPE=float32`
- `ENCODEC_LOGIT_QSTEP=1/128`
- `ENCODEC_LM_TAU=1.0`
- `ENCODEC_AC_FP_SCALE=8192`
- `ENCODEC_AC_MIN_RANGE=2`
- `ENCODEC_USE_NEAR_UNIFORM=0`

These settings performed better than the more conservative profile while keeping the `mps -> cpu` decode fix.

## Updated Measurements

### Per-clip sweep, 4 seconds, `AFTER DARK`, LM, `encodec_48khz`, `6 kbps`

- Conservative deterministic profile: `3924 bps`, encode `1.230s`, decode `1.271s`
- Lean `float32` profile: `3574 bps`, encode `1.138s`, decode `1.154s`
- Finer `float32` profile: `3556 bps`, encode `1.217s`, decode `1.339s`

Interpretation:

- The finer profile saves only a small number of bits while costing more time.
- The lean `float32` profile is the best default tradeoff so far.

### Multi-track sweep, 3 Lori EP clips, 6 seconds each, CPU, LM

- Baseline legacy branch: `4148.9 bps`, encode `1.310s`, decode `1.411s`
- Deterministic branch before tuning: `4456.4 bps`, encode `1.874s`, decode `1.932s`
- Deterministic branch with lean `float32` settings: `4227.6 bps`, encode `1.591s`, decode `1.706s`

Interpretation:

- Tuning recovered most of the bitrate loss.
- Runtime is still slower than legacy, but materially better than the earlier deterministic configuration.

### Chunk-framed `acv=4`

- Clean 4-second LM encode on `AFTER DARK`: `3654 bps`
- Cross-device `mps -> cpu`: decode succeeds
- Single flipped payload byte at mid-stream:
  - decode still succeeds
  - corrupted region spans about `0.998s`
  - rest of the file remains decodable

Interpretation:

- The extra chunk framing overhead is modest.
- We now have actual failure containment aligned with segment size, not just a design intention.

### Cross-host Mac -> Linux Decode Matrix

Using payloads encoded on the Mac and decoded on the Linode box:

- Baseline legacy payloads:
  - Mac `cpu` -> Linux `cpu`: fail with `EOFError`
  - Mac `cpu` -> Linux `cuda`: fail with `EOFError`
  - Mac `mps` -> Linux `cpu`: fail with `EOFError`
  - Mac `mps` -> Linux `cuda`: fail with `EOFError`
- Current deterministic `acv=4` payloads:
  - Mac `cpu` -> Linux `cpu`: success
  - Mac `cpu` -> Linux `cuda`: success
  - Mac `mps` -> Linux `cpu`: success
  - Mac `mps` -> Linux `cuda`: success

Interpretation:

- The deterministic entropy path is now robust across host architecture and device changes.
- Keeping the arithmetic path on CPU while allowing model decode on CUDA is the right split for cross-host determinism.

### Chunk Size Sweep

On `AFTER DARK`, 4 seconds, `encodec_48khz`, `6 kbps`, LM:

- `1.0s` chunks, Mac CPU: `3654 bps`, encode `1.328s`, decode `1.466s`
- `0.5s` chunks, Mac CPU: `4056 bps`, encode `1.192s`, decode `0.667s`
- `0.25s` chunks, Mac CPU: `4618 bps`, encode `1.330s`, decode `0.402s`
- `0.5s` chunks, Mac MPS: `4056 bps`, encode `2.730s`, decode `0.720s`

Interpretation:

- `0.25s` is too expensive in bitrate for the current design.
- `0.5s` is a plausible next point if we want stronger containment, but it costs about `+402 bps` versus `1.0s` on this clip.
- The overhead increase is dominated by segmentation behavior, not the `8-byte` per-chunk framing.

### Corruption Targeting And Current Readout

After fixing the harness to corrupt actual `acv=4` chunk bodies:

- Local Mac decode, `1.0s` chunks:
  - corruption landed in chunk index `1`
  - damaged region about `0.998s`
- Local Mac decode, `0.5s` chunks:
  - corruption landed in chunk index `4`
  - damaged region about `0.040s`

Interpretation:

- The `1.0s` result matches the design goal of containing a catastrophic failure to about one chunk.
- The `0.5s` result needs more inspection. The affected region was much smaller than the nominal chunk size, which suggests the chosen chunk landed late in the stream schedule or the actual audible effect is smaller than the full chunk substitution window.
- One unresolved inconsistency remains: the Linux clean/corrupt pair comparison for the `1.0s` corrupted payload reported zero waveform delta even though the local harness measured the expected one-chunk effect. The payload bytes are definitely different on both hosts, so this needs a targeted follow-up.

## Next Iteration Targets

1. Recover LM efficiency without losing the `mps -> cpu` fix.
2. Resolve the remaining cross-host corruption comparison inconsistency for `1.0s` chunks.
3. Decide whether `0.5s` is worth the bitrate tradeoff, or whether we should keep `1.0s` model chunks and look for finer-grained entropy chunking.
4. Reduce `acv=4` overhead by tightening chunk headers or making CRC optional.
5. Speed up chunked decode, which currently pays extra per-segment decode overhead.
6. Add automated tests for:
   - legacy decode
   - `mps -> cpu` decode
   - Mac `cpu/mps` -> Linux `cpu/cuda` decode
   - chunk corruption fallback
   - no-LM regression coverage
