# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API to compress/decompress audio to bytestreams."""

import atexit
import concurrent.futures
import io
import math
import multiprocessing
import os
import struct
import typing as tp
import zlib
from concurrent.futures.process import BrokenProcessPool

import torch

try:
    import encodec_native as _encodec_native
except ImportError:
    _encodec_native = None

try:
    from . import torch_ext as _torch_ext_loader
except Exception:
    _torch_ext_loader = None

from . import binary
from .model import EncodecModel, EncodedFrame
from .quantization.ac import (
    ArithmeticCoder,
    ArithmeticDecoder,
    build_stable_quantized_cdf,
)
from .utils import _linear_overlap_add

torch.use_deterministic_algorithms(True)
torch.backends.mkldnn.enabled = False

MODELS = {
    'encodec_24khz': EncodecModel.encodec_model_24khz,
    'encodec_48khz': EncodecModel.encodec_model_48khz,
}

# ---------------------------------------------------------------------------
# Runtime-tunable defaults via environment variables.
# Lean float32 profile (validated cross-platform: mps→cpu, cpu→cuda).
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}

def _env_dtype(name: str, default: torch.dtype) -> torch.dtype:
    v = os.getenv(name)
    if v is None:
        return default
    mapping = {"float32": torch.float32, "fp32": torch.float32,
               "float64": torch.float64, "fp64": torch.float64}
    try:
        return mapping[v.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype override {v!r} for {name}.") from exc

def _env_choice(name: str, default: str, choices: tp.Set[str]) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    value = v.lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported value {v!r} for {name}. Expected one of: {allowed}.")
    return value

# Conservative defaults: float64 LM, slightly coarser logit quantisation,
# and wider CDF bins for better cross-host determinism.
LOGIT_QSTEP          = _env_float("ENCODEC_LOGIT_QSTEP", 1.0 / 64.0)
LM_TAU               = _env_float("ENCODEC_LM_TAU", 1.0)
FP_SCALE             = _env_int("ENCODEC_AC_FP_SCALE", 1 << 13)
MIN_RANGE            = _env_int("ENCODEC_AC_MIN_RANGE", 2)
USE_NEAR_UNIFORM     = _env_bool("ENCODEC_USE_NEAR_UNIFORM", False)
DETERMINISTIC_LM_DTYPE = _env_dtype("ENCODEC_DETERMINISTIC_LM_DTYPE", torch.float64)
LM_DEVICE_MODE       = _env_choice("ENCODEC_LM_DEVICE", "cpu", {"cpu", "model"})
DECODE_LM_DEVICE_MODE = _env_choice("ENCODEC_DECODE_LM_DEVICE", "auto", {"auto", "cpu", "model"})
LM_CHUNKED_DEFAULT   = _env_bool("ENCODEC_LM_CHUNKED", True)
SEGMENT_WORKERS_DEFAULT = _env_int("ENCODEC_SEGMENT_WORKERS", 1)
NATIVE_AC_ENABLED    = _env_bool("ENCODEC_NATIVE_AC", True)
TORCH_EXT_AC_ENABLED = _env_bool("ENCODEC_TORCH_EXT", False)
ARITHMETIC_TOTAL_RANGE_BITS = 24

_IDX_CACHE: tp.Dict[tp.Tuple[str, int, int], torch.Tensor] = {}
_UNIFORM_CDF_CACHE: tp.Dict[tp.Tuple[str, int, int, int, int], torch.Tensor] = {}
_CHUNK_HEADER = struct.Struct('!II')   # chunk_len (uint32 BE), crc32 (uint32 BE)
ProgressCallback = tp.Optional[tp.Callable[[tp.Dict[str, tp.Any]], None]]
_WORKER_MODEL_CACHE: tp.Dict[tp.Tuple[str, float], EncodecModel] = {}
_WORKER_LM_CACHE: tp.Dict[tp.Tuple[str, float, str], tp.Any] = {}
# Preview/audio decode is a hot path in scratch.fm, so keep decoder models and
# LM instances alive instead of rebuilding them for every payload.
_DECODE_MODEL_CACHE: tp.Dict[tp.Tuple[str, str], EncodecModel] = {}
_DECODE_LM_CACHE: tp.Dict[tp.Tuple[str, str, str, float], tp.Any] = {}
_DECODE_LEGACY_LM_CACHE: tp.Dict[tp.Tuple[str, str, str], tp.Any] = {}
_PARALLEL_EXECUTOR: tp.Optional[concurrent.futures.ProcessPoolExecutor] = None
_PARALLEL_EXECUTOR_WORKERS = 0
_TORCH_AC_MODULE: tp.Optional[tp.Any] = None
_TORCH_AC_LOAD_FAILED = False


# ---------------------------------------------------------------------------
# CDF / probability helpers
# ---------------------------------------------------------------------------

def _counts_from_pdf(pdf: torch.Tensor, fp_scale: int) -> torch.Tensor:
    """Convert a PDF to integer counts via floor(pdf * fp_scale) in float64.

    Near-integer fractions receive a deterministic ±ε perturbation to break
    ties consistently across platforms.  The result is clamped to ≥0 so that
    exact-zero probabilities (common at tau=1.0 due to float underflow of
    exp(-large)) never produce −1 via floor(0 − ε).
    """
    x = (pdf.detach().to(torch.float64).clamp_min(0) * fp_scale)
    fx = torch.floor(x)
    frac = x - fx
    eps_edge = math.ldexp(1.0, -40)
    m = (frac <= eps_edge) | (frac >= 1 - eps_edge)
    if bool(m.any()):
        idx = torch.arange(x.numel(), device=x.device, dtype=torch.int64).view_as(x)
        sign = (idx.fmod(2) * 2 - 1).to(torch.float64)
        eps = math.ldexp(1.0, -60)
        x = torch.where(m, x + sign * eps, x)
        # clamp before floor: negative sign on an exact-zero pdf would give
        # x = −ε → floor = −1, corrupting the CDF.
        fx = torch.floor(x.clamp_min(0))
    return fx.to(torch.int64)


def _quantize_logits_(logits: torch.Tensor, step: float = LOGIT_QSTEP) -> torch.Tensor:
    """Round logits to a deterministic grid (biased-floor half-step)."""
    y = (logits / step).to(torch.float64)
    eps = math.ldexp(1.0, -40)
    q = torch.floor(y + 0.5 - eps)
    return q * step


def _stable_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax in float64 using a sequential cumsum denominator for
    cross-architecture bit-reproducibility."""
    x = (logits - torch.amax(logits, dim=dim, keepdim=True)).to(torch.float64)
    z = torch.exp(x)
    z = z.movedim(dim, -1).contiguous()
    acc = torch.cumsum(z, dim=-1)[..., -1]
    return (z / acc.unsqueeze(-1)).movedim(-1, dim)


def _softmax_or_uniform(x: torch.Tensor, dim: int) -> torch.Tensor:
    s = _stable_softmax(x, dim)
    if not USE_NEAR_UNIFORM:
        return s
    span_logit = torch.amax(x, dim=dim, keepdim=True) - torch.amin(x, dim=dim, keepdim=True)
    near_logit = span_logit <= (2 * LOGIT_QSTEP)
    span_pdf = torch.amax(s, dim=dim, keepdim=True) - torch.amin(s, dim=dim, keepdim=True)
    near_pdf = span_pdf <= (0.25 / float(FP_SCALE))
    near = near_logit | near_pdf
    if not bool(near.any()):
        return s
    k = x.size(dim)
    u = torch.full_like(s, 1.0 / k, dtype=torch.float64)
    return torch.where(near, u, s)


def _torch_ac_module() -> tp.Optional[tp.Any]:
    global _TORCH_AC_MODULE, _TORCH_AC_LOAD_FAILED
    if not TORCH_EXT_AC_ENABLED or _torch_ext_loader is None or _TORCH_AC_LOAD_FAILED:
        return None
    if _TORCH_AC_MODULE is not None:
        return _TORCH_AC_MODULE
    try:
        _TORCH_AC_MODULE = _torch_ext_loader.load_extension()
    except Exception:
        _TORCH_AC_LOAD_FAILED = True
        return None
    return _TORCH_AC_MODULE


def _tensor_native_ac_module() -> tp.Optional[tp.Any]:
    module = _torch_ac_module()
    if module is not None:
        return module
    if NATIVE_AC_ENABLED and _encodec_native is not None:
        return _encodec_native
    return None


def _native_ac_available() -> bool:
    return _tensor_native_ac_module() is not None


def _can_batch_lm_encode(lm_device: torch.device, coder_device: torch.device) -> bool:
    # Only batch the deterministic CPU path that we have validated byte-for-byte
    # against the existing stepwise encoder.
    return lm_device.type == "cpu" and coder_device.type == "cpu"


def _compute_lm_probas_for_frame(
    frame: torch.Tensor,
    *,
    lm: tp.Any,
    lm_device: torch.device,
    lm_tau: float,
) -> torch.Tensor:
    """Run the LM over a whole frame with teacher forcing.

    The returned probabilities are shaped [1, card, K, T] and match the
    stepwise encoder's quantized CDFs on the deterministic CPU path.
    """
    _B, K, T = frame.shape
    if T <= 0:
        raise ValueError("LM frame must contain at least one timestep.")

    prefix = torch.zeros(1, K, 1, dtype=torch.long, device=lm_device)
    if T == 1:
        teacher = prefix
    else:
        teacher = torch.cat([prefix, 1 + frame[:, :, :-1].detach().to(lm_device)], dim=-1)

    with torch.inference_mode():
        logits_raw, _, _ = lm.forward_logits(teacher, None, 0)
        logits_q = _quantize_logits_(logits_raw / lm_tau, LOGIT_QSTEP)
        return _softmax_or_uniform(logits_q, dim=1)


def _flatten_lm_block_for_coder(
    probas: torch.Tensor,
    frame: torch.Tensor,
    *,
    coder_device: torch.device,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Flatten a full LM block into time-major columns for the entropy coder."""
    pdf_cols = (
        probas[0]
        .permute(0, 2, 1)
        .contiguous()
        .reshape(probas.shape[1], -1)
        .to(coder_device)
    )
    symbols = (
        frame[0]
        .transpose(0, 1)
        .contiguous()
        .reshape(-1)
        .detach()
        .to(coder_device)
    )
    return pdf_cols, symbols


def _deterministic_cdf(pdf: torch.Tensor,
                       total_range_bits: int,
                       fp_scale: int = FP_SCALE,
                       min_range: int = MIN_RANGE,
                       check: bool = False) -> torch.Tensor:
    """Architecture-stable integer CDF for a single PDF vector."""
    pdf = pdf.detach().to(torch.float64).clamp_min(0)
    s = pdf.sum()
    if (not torch.isfinite(s)) or (s <= 0):
        pdf = torch.ones_like(pdf)

    num = _counts_from_pdf(pdf, fp_scale).to(torch.int64)
    if int(num.sum().item()) <= 0:
        num = torch.ones_like(num)

    total = 1 << total_range_bits
    n = int(num.numel())
    alloc = total - min_range * n
    num_sum = int(num.sum().item())

    base = (alloc * num) // num_sum
    remainder = int(alloc - int(base.sum().item()))
    if remainder > 0:
        prio = (alloc * num) - (num_sum * base)
        idx = torch.arange(n, device=num.device, dtype=torch.int64)
        key = prio * (n + 1) - idx
        _, order = torch.sort(key, descending=True, stable=True)
        base[order[:remainder]] += 1

    ranges = base + min_range
    cdf = torch.cumsum(ranges, dim=-1, dtype=torch.int64)
    if check:
        assert int(cdf[-1].item()) == total
        assert (ranges >= min_range).all()
    return cdf


def _deterministic_cdf_multi(pdf_mat: torch.Tensor,
                              total_range_bits: int,
                              fp_scale: int = FP_SCALE,
                              min_range: int = MIN_RANGE,
                              check: bool = False) -> torch.Tensor:
    """Vectorised _deterministic_cdf over [bins, K] PDF matrix."""
    assert pdf_mat.dim() == 2, "pdf_mat must be 2D: [bins, K]"
    pdf = pdf_mat.detach().to(torch.float64).clamp_min(0)
    s = torch.sum(pdf, dim=0)
    invalid = (~torch.isfinite(s)) | (s <= 0)
    if bool(invalid.any()):
        pdf[:, invalid] = 1.0

    # Shortcut: detect fully-uniform columns and cache their CDF.
    eq0 = (pdf[0:1, :] == pdf)
    uniform_mask = torch.all(eq0, dim=0)

    num = _counts_from_pdf(pdf, fp_scale).to(torch.int64)
    zeros = torch.sum(num, dim=0) <= 0
    if bool(zeros.any()):
        num[:, zeros] = 1

    total = 1 << total_range_bits
    n_bins = int(num.size(0))
    alloc = total - min_range * n_bins
    num_sum = torch.sum(num, dim=0)

    base = (alloc * num) // num_sum
    base_sum = torch.sum(base, dim=0)
    remainder = (alloc - base_sum).to(torch.int64)

    if bool((remainder > 0).any()):
        prio = (alloc * num) - (num_sum * base)
        dev = num.device
        dev_key = (dev.type, -1 if dev.index is None else int(dev.index), n_bins)
        idx_row = _IDX_CACHE.get(dev_key)
        if idx_row is None:
            idx_row = torch.arange(n_bins, device=dev, dtype=torch.int64).unsqueeze(1)
            _IDX_CACHE[dev_key] = idx_row
        idx = idx_row.expand(n_bins, num.size(1))
        key = prio * (n_bins + 1) - idx
        order = torch.argsort(key, dim=0, descending=True, stable=True)
        max_rem = int(torch.max(remainder).item())
        if max_rem > 0:
            top_idx = order[:max_rem, :]
            row_range = torch.arange(max_rem, device=num.device, dtype=torch.int64).unsqueeze(1)
            take_mask = (row_range < remainder.unsqueeze(0)).to(base.dtype)
            base = base.scatter_add(0, top_idx, take_mask)

    ranges = base + min_range
    cdf = torch.cumsum(ranges, dim=0, dtype=torch.int64)

    if bool(uniform_mask.any()):
        dev = pdf.device
        cache_key = (dev.type, -1 if dev.index is None else int(dev.index),
                     n_bins, int(total_range_bits), int(min_range))
        u_cdf = _UNIFORM_CDF_CACHE.get(cache_key)
        if u_cdf is None:
            u_pdf = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float64, device=dev)
            u_cdf = _deterministic_cdf(u_pdf, total_range_bits,
                                       fp_scale=fp_scale, min_range=min_range)
            _UNIFORM_CDF_CACHE[cache_key] = u_cdf
        cdf[:, uniform_mask] = u_cdf.unsqueeze(1)

    if check:
        assert torch.all(cdf[-1, :] == total)
        assert torch.all(ranges >= min_range)
    return cdf


# ---------------------------------------------------------------------------
# acv=4 chunk framing helpers
# ---------------------------------------------------------------------------

def _emit_progress(progress_callback: ProgressCallback, payload: tp.Dict[str, tp.Any]) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:
        # Progress reporting must never affect deterministic bytestream generation.
        pass


def _segment_layout(model: EncodecModel, audio_length: int) -> tp.Tuple[int, int, tp.List[int]]:
    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length
    offsets = list(range(0, audio_length, segment_stride))
    return segment_length, segment_stride, offsets


def _build_progress_payload(
    *,
    stage: str,
    sample_rate: int,
    total_segments: int,
    segment_index: int,
    audio_length: int,
    segment_length: int,
    segment_stride: int,
    offset_samples: int = 0,
) -> tp.Dict[str, tp.Any]:
    payload: tp.Dict[str, tp.Any] = {
        'stage': stage,
        'segmentCount': total_segments,
        'segmentIndex': segment_index,
        'progress': float(segment_index / total_segments) if total_segments else 0.0,
        'sampleRate': int(sample_rate),
        'audioLength': audio_length,
        'segmentLength': int(segment_length),
        'segmentStride': int(segment_stride),
    }
    if stage == 'segment':
        payload['offsetSamples'] = int(offset_samples)
    return payload


def _parallel_segment_worker_count(
    total_segments: int,
    *,
    use_lm: bool,
    lm_chunked: bool,
    model_device: torch.device,
) -> int:
    configured = SEGMENT_WORKERS_DEFAULT
    if configured <= 0:
        configured = os.cpu_count() or 1
    if (
        configured <= 1
        or total_segments <= 1
        or not use_lm
        or not lm_chunked
        or model_device.type != 'cpu'
        or LM_DEVICE_MODE != 'cpu'
    ):
        return 1
    return max(1, min(int(configured), int(total_segments)))


def _build_segment_batches(
    wav: torch.Tensor,
    offsets: tp.List[int],
    segment_length: int,
    worker_count: int,
) -> tp.List[tp.List[tp.Tuple[int, int, torch.Tensor]]]:
    batch_count = max(1, min(worker_count, len(offsets)))
    batch_size = int(math.ceil(len(offsets) / batch_count))
    batches: tp.List[tp.List[tp.Tuple[int, int, torch.Tensor]]] = []
    for start in range(0, len(offsets), batch_size):
        batch: tp.List[tp.Tuple[int, int, torch.Tensor]] = []
        for absolute_index, offset_samples in enumerate(offsets[start:start + batch_size], start=start + 1):
            segment = wav[:, offset_samples: offset_samples + segment_length].detach().cpu().contiguous()
            batch.append((absolute_index, int(offset_samples), segment))
        batches.append(batch)
    return batches


def _init_parallel_worker_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def _shutdown_parallel_executor() -> None:
    global _PARALLEL_EXECUTOR
    global _PARALLEL_EXECUTOR_WORKERS
    executor = _PARALLEL_EXECUTOR
    _PARALLEL_EXECUTOR = None
    _PARALLEL_EXECUTOR_WORKERS = 0
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def _get_parallel_executor(worker_count: int) -> concurrent.futures.ProcessPoolExecutor:
    global _PARALLEL_EXECUTOR
    global _PARALLEL_EXECUTOR_WORKERS
    if worker_count <= 1:
        raise ValueError('worker_count must be greater than 1 for the parallel executor.')
    if _PARALLEL_EXECUTOR is None or _PARALLEL_EXECUTOR_WORKERS != worker_count:
        _shutdown_parallel_executor()
        _PARALLEL_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_parallel_worker_runtime,
        )
        _PARALLEL_EXECUTOR_WORKERS = worker_count
    return _PARALLEL_EXECUTOR
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _get_parallel_worker_model(
    model_name: str,
    bandwidth: float,
    *,
    use_lm: bool,
    lm_tau: float,
) -> tp.Tuple[EncodecModel, tp.Optional[tp.Any]]:
    model_key = (model_name, float(bandwidth))
    model = _WORKER_MODEL_CACHE.get(model_key)
    if model is None:
        model = MODELS[model_name]().eval()
        model.set_target_bandwidth(float(bandwidth))
        model.to('cpu')
        _WORKER_MODEL_CACHE[model_key] = model

    lm = None
    if use_lm:
        lm_key = (model_name, float(bandwidth), str(DETERMINISTIC_LM_DTYPE))
        lm = _WORKER_LM_CACHE.get(lm_key)
        if lm is None:
            lm = model.get_lm_model(
                device=torch.device('cpu'),
                dtype=DETERMINISTIC_LM_DTYPE,
            ).eval()
            _WORKER_LM_CACHE[lm_key] = lm
        lm.tau = float(lm_tau)

    return model, lm


def _device_key(device: tp.Union[str, torch.device]) -> str:
    return str(torch.device(device))


def _get_decode_model(model_name: str, device: tp.Union[str, torch.device]) -> EncodecModel:
    key = (model_name, _device_key(device))
    model = _DECODE_MODEL_CACHE.get(key)
    if model is None:
        model = MODELS[model_name]().to(device).eval()
        _DECODE_MODEL_CACHE[key] = model
    return model


def _select_decode_lm_device(
    *,
    model_device: tp.Union[str, torch.device],
    coder_device: tp.Union[str, torch.device],
    acv: int,
) -> torch.device:
    model_device = torch.device(model_device)
    coder_device = torch.device(coder_device)

    if acv < 3:
        return coder_device
    if DECODE_LM_DEVICE_MODE == "cpu":
        return coder_device
    if DECODE_LM_DEVICE_MODE == "model":
        return model_device
    # Auto: keep legacy / CPU-safe behavior everywhere except CUDA, where the
    # deterministic float64 LM path is materially faster and parity-clean.
    if model_device.type == "cuda":
        return model_device
    return coder_device


def _get_decode_lms(
    model: EncodecModel,
    *,
    model_name: str,
    coder_device: tp.Union[str, torch.device],
    lm_device: tp.Union[str, torch.device],
    use_lm: bool,
    acv: int,
    lm_tau: float,
) -> tp.Tuple[tp.Optional[tp.Any], tp.Optional[tp.Any]]:
    coder_key = _device_key(coder_device)
    if not use_lm:
        return None, None

    if acv >= 3:
        lm_key = (model_name, _device_key(lm_device), str(DETERMINISTIC_LM_DTYPE), float(lm_tau))
        lm = _DECODE_LM_CACHE.get(lm_key)
        if lm is None:
            lm = model.get_lm_model(
                device=torch.device(lm_device),
                dtype=DETERMINISTIC_LM_DTYPE,
            ).eval()
            lm.tau = float(lm_tau)
            _DECODE_LM_CACHE[lm_key] = lm
        return lm, None

    legacy_key = (model_name, coder_key, str(torch.float32))
    legacy_lm = _DECODE_LEGACY_LM_CACHE.get(legacy_key)
    if legacy_lm is None:
        legacy_lm = model.get_lm_model(
            device=torch.device(coder_key),
            dtype=torch.float32,
        ).eval()
        _DECODE_LEGACY_LM_CACHE[legacy_key] = legacy_lm
    return None, legacy_lm


def _encode_segment_batch_worker(
    model_name: str,
    bandwidth: float,
    use_lm: bool,
    lm_tau: float,
    batch: tp.List[tp.Tuple[int, int, torch.Tensor]],
) -> dict:
    _init_parallel_worker_runtime()
    model, lm = _get_parallel_worker_model(
        model_name,
        bandwidth,
        use_lm=use_lm,
        lm_tau=lm_tau,
    )
    coder_device = torch.device('cpu')
    lm_device = torch.device('cpu')
    segments: tp.List[tp.Tuple[int, int, bytes]] = []
    num_codebooks: tp.Optional[int] = None

    for segment_index, offset_samples, segment in batch:
        segment_wav = segment.unsqueeze(0)
        with torch.inference_mode():
            frame, scale = model._encode_frame(segment_wav.to(coder_device))
        if num_codebooks is None:
            num_codebooks = int(frame.shape[1])

        payload_fo = io.BytesIO()
        _write_frame_payload(
            frame,
            scale,
            payload_fo,
            use_lm=use_lm,
            model=model,
            coder_device=coder_device,
            lm_device=lm_device,
            lm=lm,
            lm_tau=lm_tau,
        )

        framed_fo = io.BytesIO()
        if use_lm:
            _write_chunk(framed_fo, payload_fo.getvalue())
        else:
            framed_fo.write(payload_fo.getvalue())
        segments.append((int(segment_index), int(offset_samples), framed_fo.getvalue()))

    return {
        'numCodebooks': int(num_codebooks or 0),
        'segments': segments,
    }


atexit.register(_shutdown_parallel_executor)

def _write_chunk(fo: tp.IO[bytes], payload: bytes) -> None:
    """Write a CRC-protected chunk: [len: u32][crc: u32][payload]."""
    fo.write(_CHUNK_HEADER.pack(len(payload), zlib.crc32(payload) & 0xffffffff))
    fo.write(payload)


def _read_chunk_payload(fo: tp.IO[bytes]) -> bytes:
    """Read and CRC-verify one chunk.  Raises ValueError on mismatch."""
    chunk_len, chunk_crc = _CHUNK_HEADER.unpack(binary._read_exactly(fo, _CHUNK_HEADER.size))
    payload = binary._read_exactly(fo, chunk_len)
    actual = zlib.crc32(payload) & 0xffffffff
    if actual != chunk_crc:
        raise ValueError(f"Chunk CRC mismatch: expected {chunk_crc:#010x}, got {actual:#010x}.")
    return payload


# ---------------------------------------------------------------------------
# compress_to_file / decompress_from_file
# ---------------------------------------------------------------------------

def _write_frame_payload(
    frame: torch.Tensor,
    scale: tp.Optional[torch.Tensor],
    fo: tp.IO[bytes],
    *,
    use_lm: bool,
    model: EncodecModel,
    coder_device: torch.device,
    lm_device: torch.device,
    lm: tp.Optional[tp.Any],
    lm_tau: float,
) -> None:
    if scale is not None:
        fo.write(struct.pack('!f', float(scale.cpu().item())))

    _B, K, T = frame.shape
    if use_lm:
        assert lm is not None
        native_coder = None
        coder = None
        native_module = _tensor_native_ac_module()
        if native_module is not None:
            native_coder = native_module.ArithmeticEncoder(ARITHMETIC_TOTAL_RANGE_BITS)
        else:
            coder = ArithmeticCoder(fo, total_range_bits=ARITHMETIC_TOTAL_RANGE_BITS)
        if _can_batch_lm_encode(lm_device, coder_device):
            probas = _compute_lm_probas_for_frame(
                frame,
                lm=lm,
                lm_device=lm_device,
                lm_tau=lm_tau,
            )
            pdf_cols, symbol_tensor = _flatten_lm_block_for_coder(
                probas,
                frame,
                coder_device=coder_device,
            )
            if native_coder is not None:
                native_coder.push_pdf_symbols_torch(
                    pdf_cols.detach().contiguous(),
                    symbol_tensor.detach().contiguous(),
                    FP_SCALE,
                    MIN_RANGE,
                )
            else:
                assert coder is not None
                cdf_mat = _deterministic_cdf_multi(
                    pdf_cols, coder.total_range_bits,
                    fp_scale=FP_SCALE, min_range=MIN_RANGE, check=False)
                cdf_cols = cdf_mat.t().contiguous()
                for col, value in enumerate(symbol_tensor.tolist()):
                    coder.push(value, cdf_cols[col])
            if native_coder is not None:
                fo.write(bytes(native_coder.finish()))
            else:
                assert coder is not None
                coder.flush()
            return
        states = None
        offset = 0
        input_ = torch.zeros(1, K, 1, dtype=torch.long, device=lm_device)
    else:
        packer = binary.BitPacker(model.bits_per_codebook, fo)

    for t in range(T):
        if use_lm:
            with torch.inference_mode():
                logits_raw, states, offset = lm.forward_logits(input_, states, offset)
                logits_q = _quantize_logits_(logits_raw / lm_tau, LOGIT_QSTEP)
                probas = _softmax_or_uniform(logits_q, dim=1)

            pdf_mat = probas[0, :, :, 0].to(coder_device)
            frame_slice = frame[:, :, t:t + 1].detach().to(coder_device)
            symbol_tensor = frame_slice[0, :, 0].detach().contiguous()
            if native_coder is not None:
                native_coder.push_pdf_symbols_torch(
                    pdf_mat.detach().contiguous(),
                    symbol_tensor,
                    FP_SCALE,
                    MIN_RANGE,
                )
            else:
                assert coder is not None
                cdf_mat = _deterministic_cdf_multi(
                    pdf_mat, coder.total_range_bits,
                    fp_scale=FP_SCALE, min_range=MIN_RANGE, check=False)
                cdf_cols = cdf_mat.t().contiguous()
                for k, value in enumerate(symbol_tensor.tolist()):
                    coder.push(value, cdf_cols[k])
            input_ = (1 + frame_slice).to(lm_device)
        else:
            for value in frame[0, :, t].detach().cpu().tolist():
                packer.push(value)

    if use_lm:
        if native_coder is not None:
            fo.write(bytes(native_coder.finish()))
        else:
            assert coder is not None
            coder.flush()
    else:
        packer.flush()


def compress_to_file(model: EncodecModel, wav: torch.Tensor, fo: tp.IO[bytes],
                     use_lm: bool = True,
                     progress_callback: ProgressCallback = None,
                     lm_chunked: tp.Optional[bool] = None) -> None:
    """Compress a waveform to a file-object.

    When ``use_lm=True``:
      * ``lm_chunked=True`` writes bitstream version 4 (acv=4), where
        each model segment is wrapped in a CRC-protected chunk.
      * ``lm_chunked=False`` writes deterministic unchunked bitstream
        version 3 (acv=3), compatible with the existing deterministic
        decoder path.

    The arithmetic coder and LM always run on CPU for cross-platform
    determinism unless ``ENCODEC_LM_DEVICE=model`` is set. The EnCodec
    model itself may run on any device.

    Args:
        model: pre-trained EncodecModel.
        wav: ``[C, T]`` waveform at model.sample_rate.
        fo: writable file-object.
        use_lm: enable LM entropy coding.
        lm_chunked: choose CRC chunk framing for deterministic LM streams.
    """
    assert wav.dim() == 2
    if model.name not in MODELS:
        raise ValueError(f"Unsupported model {model.name}.")

    if lm_chunked is None:
        lm_chunked = LM_CHUNKED_DEFAULT

    model = model.eval()
    model_device = next(model.parameters()).device
    coder_device = torch.device("cpu")
    lm_device = model_device if LM_DEVICE_MODE == "model" else coder_device
    audio_length = int(wav.shape[-1])
    segment_length, segment_stride, offsets = _segment_layout(model, audio_length)

    if not offsets:
        raise ValueError("Cannot compress an empty waveform.")

    lm = None
    lm_tau = LM_TAU
    total_segments = len(offsets)
    _emit_progress(progress_callback, _build_progress_payload(
        stage='start',
        sample_rate=int(model.sample_rate),
        total_segments=total_segments,
        segment_index=0,
        audio_length=audio_length,
        segment_length=segment_length,
        segment_stride=segment_stride,
    ))

    if use_lm and not lm_chunked:
        with torch.inference_mode():
            frames = model.encode(wav[None].to(model_device))
        if not frames:
            raise ValueError("Cannot compress an empty waveform.")

        codes0, _ = frames[0]
        _, K, _ = codes0.shape
        lm = model.get_lm_model(device=lm_device, dtype=DETERMINISTIC_LM_DTYPE).eval()
        lm.tau = lm_tau
        metadata: tp.Dict[str, tp.Any] = {
            'm':   model.name,
            'al':  audio_length,
            'nc':  int(K),
            'lm':  True,
            'fp':  int(FP_SCALE),
            'mr':  int(MIN_RANGE),
            'acv': 3,
            'tau': float(lm_tau),
        }
        binary.write_ecdc_header(fo, metadata)

        for segment_index, ((frame, scale), offset_samples) in enumerate(zip(frames, offsets), start=1):
            _write_frame_payload(
                frame,
                scale,
                fo,
                use_lm=True,
                model=model,
                coder_device=coder_device,
                lm_device=lm_device,
                lm=lm,
                lm_tau=lm_tau,
            )
            _emit_progress(progress_callback, _build_progress_payload(
                stage='segment',
                sample_rate=int(model.sample_rate),
                total_segments=total_segments,
                segment_index=segment_index,
                audio_length=audio_length,
                segment_length=segment_length,
                segment_stride=segment_stride,
                offset_samples=int(offset_samples),
            ))
        return

    parallel_workers = _parallel_segment_worker_count(
        total_segments,
        use_lm=use_lm,
        lm_chunked=bool(lm_chunked),
        model_device=model_device,
    )

    if parallel_workers > 1:
        num_codebooks = int(model.quantizer.get_num_quantizers_for_bandwidth(
            model.frame_rate,
            model.bandwidth,
        ))
        metadata = {
            'm':   model.name,
            'al':  audio_length,
            'nc':  num_codebooks,
            'lm':  bool(use_lm),
            'fp':  int(FP_SCALE),
            'mr':  int(MIN_RANGE),
            'acv': 4 if use_lm else 0,
            'tau': float(lm_tau),
        }
        binary.write_ecdc_header(fo, metadata)

        batches = _build_segment_batches(wav, offsets, segment_length, parallel_workers)
        completed_segments = 0
        ordered_results: tp.List[dict] = []
        executor = _get_parallel_executor(parallel_workers)
        try:
            futures = [
                executor.submit(
                    _encode_segment_batch_worker,
                    model.name,
                    float(model.bandwidth or 0.0),
                    bool(use_lm),
                    float(lm_tau),
                    batch,
                )
                for batch in batches
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                ordered_results.append(result)
                completed_segments += len(result['segments'])
                last_index, last_offset, _ = result['segments'][-1]
                _emit_progress(progress_callback, _build_progress_payload(
                    stage='segment',
                    sample_rate=int(model.sample_rate),
                    total_segments=total_segments,
                    segment_index=min(completed_segments, total_segments),
                    audio_length=audio_length,
                    segment_length=segment_length,
                    segment_stride=segment_stride,
                    offset_samples=int(last_offset),
                ))
        except BrokenProcessPool:
            _shutdown_parallel_executor()
            raise

        for result in sorted(ordered_results, key=lambda item: item['segments'][0][0]):
            for _, _, framed_payload in result['segments']:
                fo.write(framed_payload)
        return

    header_written = False
    for segment_index, offset_samples in enumerate(offsets, start=1):
        with torch.inference_mode():
            segment_wav = wav[None, :, offset_samples: offset_samples + segment_length].to(model_device)
            frame, scale = model._encode_frame(segment_wav)

        if not header_written:
            _, K, _ = frame.shape
            if use_lm:
                lm = model.get_lm_model(device=lm_device,
                                        dtype=DETERMINISTIC_LM_DTYPE).eval()
                lm.tau = lm_tau

            metadata = {
                'm':   model.name,
                'al':  audio_length,
                'nc':  int(K),
                'lm':  bool(use_lm),
                'fp':  int(FP_SCALE),
                'mr':  int(MIN_RANGE),
                'acv': 4 if use_lm else 0,
                'tau': float(lm_tau),
            }
            binary.write_ecdc_header(fo, metadata)
            header_written = True

        if use_lm:
            chunk_fo = io.BytesIO()
            _write_frame_payload(
                frame,
                scale,
                chunk_fo,
                use_lm=True,
                model=model,
                coder_device=coder_device,
                lm_device=lm_device,
                lm=lm,
                lm_tau=lm_tau,
            )
            _write_chunk(fo, chunk_fo.getvalue())
        else:
            _write_frame_payload(
                frame,
                scale,
                fo,
                use_lm=False,
                model=model,
                coder_device=coder_device,
                lm_device=lm_device,
                lm=None,
                lm_tau=lm_tau,
            )

        _emit_progress(progress_callback, _build_progress_payload(
            stage='segment',
            sample_rate=int(model.sample_rate),
            total_segments=total_segments,
            segment_index=segment_index,
            audio_length=audio_length,
            segment_length=segment_length,
            segment_stride=segment_stride,
            offset_samples=int(offset_samples),
        ))


def decompress_from_file(fo: tp.IO[bytes],
                         device: str = 'cpu') -> tp.Tuple[torch.Tensor, int]:
    """Decompress from a file-object.  Returns ``(wav, sample_rate)``.

    Supports:
      * acv=0  — raw bitpacking (no LM).
      * acv<3  — legacy LM streams from the original Facebook implementation.
      * acv=4  — deterministic LM streams (this implementation).
                 Corrupt segments fall back to silence rather than aborting.

    The model (EnCodec encoder/decoder) runs on ``device``. The arithmetic
    coder always runs on CPU; the deterministic LM path can run on the model
    device when configured.
    """
    metadata = binary.read_ecdc_header(fo)
    model_name   = metadata['m']
    audio_length = int(metadata['al'])
    num_codebooks = int(metadata['nc'])
    use_lm       = bool(metadata['lm'])
    fp_scale     = int(metadata.get('fp', FP_SCALE))
    min_range    = int(metadata.get('mr', MIN_RANGE))
    acv          = int(metadata.get('acv', 0))
    # tau is stored since this merged implementation; fall back to env-var default
    # so we can also decode payloads from the earlier codex-precision branch.
    lm_tau       = float(metadata.get('tau', LM_TAU))

    if model_name not in MODELS:
        raise ValueError(f"Unsupported model {model_name}.")
    if acv > 4:
        raise ValueError(f"Unsupported bitstream version {acv}; re-encode.")

    model = _get_decode_model(model_name, device)
    model_device = next(model.parameters()).device
    coder_device = torch.device("cpu")
    lm_device = _select_decode_lm_device(
        model_device=model_device,
        coder_device=coder_device,
        acv=acv,
    )

    lm, legacy_lm = _get_decode_lms(
        model,
        model_name=model_name,
        coder_device=coder_device,
        lm_device=lm_device,
        use_lm=use_lm,
        acv=acv,
        lm_tau=lm_tau,
    )

    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length
    decoded_frames: tp.List[torch.Tensor] = []
    frames: tp.List[EncodedFrame] = []

    for offset_samples in range(0, audio_length, segment_stride):
        this_len = min(audio_length - offset_samples, segment_length)
        frame_length = int(math.ceil(this_len * model.frame_rate / model.sample_rate))
        frame_fo = fo

        if acv == 4:
            try:
                frame_fo = io.BytesIO(_read_chunk_payload(fo))
            except Exception:
                # Corrupt chunk → substitute silence and continue.
                decoded_frames.append(
                    torch.zeros(1, model.channels, this_len, device=model_device))
                continue

        if model.normalize:
            scale_f, = struct.unpack('!f', binary._read_exactly(
                frame_fo, struct.calcsize('!f')))
            scale = torch.tensor(scale_f, device=coder_device).view(1)
        else:
            scale = None

        if use_lm:
            native_decoder = None
            code_buf = None
            decoder = None
            native_module = _tensor_native_ac_module() if acv == 4 else None
            if native_module is not None:
                native_decoder = native_module.ArithmeticDecoder(
                    frame_fo.read(),
                    ARITHMETIC_TOTAL_RANGE_BITS,
                )
                code_buf = torch.empty(num_codebooks, dtype=torch.long, device=coder_device)
            else:
                decoder = ArithmeticDecoder(frame_fo, total_range_bits=ARITHMETIC_TOTAL_RANGE_BITS)
            states = None
            offset = 0
            input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long,
                                 device=lm_device if acv >= 3 else coder_device)
        else:
            unpacker = binary.BitUnpacker(model.bits_per_codebook, frame_fo)

        frame = torch.zeros(1, num_codebooks, frame_length,
                            dtype=torch.long, device=coder_device)
        try:
            with torch.inference_mode():
                for t in range(frame_length):
                    if use_lm and acv >= 3:
                        logits_raw, states, offset = lm.forward_logits(
                            input_, states, offset)
                        logits_q = _quantize_logits_(logits_raw / lm_tau,
                                                     LOGIT_QSTEP)
                        probas = _softmax_or_uniform(logits_q, dim=1)
                        pdf_mat = probas[0, :, :, 0].to(coder_device)
                        if native_decoder is not None:
                            assert code_buf is not None
                            native_decoder.pull_symbols_into_torch(
                                pdf_mat.detach().contiguous(),
                                code_buf,
                                fp_scale,
                                min_range,
                            )
                            frame[0, :, t] = code_buf
                            input_ = 1 + code_buf.view(1, num_codebooks, 1).to(
                                lm_device,
                            )
                        else:
                            assert decoder is not None
                            cdf_mat = _deterministic_cdf_multi(
                                pdf_mat, decoder.total_range_bits,
                                fp_scale=fp_scale, min_range=min_range, check=False)
                            cdf_cols = cdf_mat.t().contiguous()
                            code_list = []
                            for k in range(num_codebooks):
                                code = decoder.pull(cdf_cols[k])
                                if code is None:
                                    raise EOFError("Stream ended before expected.")
                                code_list.append(code)
                            frame[0, :, t] = torch.tensor(code_list, dtype=torch.long,
                                                          device=coder_device)
                            input_ = (1 + frame[:, :, t:t + 1]).to(lm_device)

                    elif use_lm:  # legacy path
                        probas, states, offset = legacy_lm.forward_legacy(
                            input_, states, offset)
                        code_list = []
                        for k in range(num_codebooks):
                            q_cdf = build_stable_quantized_cdf(
                                probas[0, :, k, 0], decoder.total_range_bits,
                                check=False)
                            code = decoder.pull(q_cdf)
                            if code is None:
                                raise EOFError("Stream ended before expected.")
                            code_list.append(code)
                        frame[0, :, t] = torch.tensor(code_list, dtype=torch.long,
                                                      device=coder_device)
                        input_ = 1 + frame[:, :, t:t + 1]

                    else:
                        code_list = []
                        for _ in range(num_codebooks):
                            code = unpacker.pull()
                            if code is None:
                                raise EOFError("Stream ended before expected.")
                            code_list.append(code)
                        frame[0, :, t] = torch.tensor(code_list, dtype=torch.long,
                                                      device=coder_device)

        except Exception:
            if acv == 4:
                decoded_frames.append(
                    torch.zeros(1, model.channels, this_len, device=model_device))
                continue
            raise

        encoded_frame = (frame.to(model_device),
                         None if scale is None else scale.to(model_device))
        if acv == 4:
            with torch.inference_mode():
                decoded_frames.append(
                    model._decode_frame(encoded_frame)[..., :this_len])
        else:
            frames.append(encoded_frame)

    if acv == 4:
        if model.segment_length is None:
            wav = decoded_frames[0]
        else:
            wav = _linear_overlap_add(decoded_frames, model.segment_stride or 1)
    else:
        with torch.inference_mode():
            wav = model.decode(frames)
    return wav[0, :, :audio_length], model.sample_rate


def compress(model: EncodecModel, wav: torch.Tensor,
             use_lm: bool = False,
             progress_callback: ProgressCallback = None,
             lm_chunked: tp.Optional[bool] = None) -> bytes:
    """Compress a waveform and return bytes."""
    fo = io.BytesIO()
    compress_to_file(
        model,
        wav,
        fo,
        use_lm=use_lm,
        progress_callback=progress_callback,
        lm_chunked=lm_chunked,
    )
    return fo.getvalue()


def decompress(compressed: bytes,
               device: str = 'cpu') -> tp.Tuple[torch.Tensor, int]:
    """Decompress from bytes.  Returns ``(wav, sample_rate)``."""
    return decompress_from_file(io.BytesIO(compressed), device=device)


def test():
    import soundfile as sf
    import time
    torch.set_num_threads(1)
    for name in MODELS.keys():
        model = MODELS[name]()
        suffix = name.split('_')[1][:3]
        x, sr = sf.read(f'test_{suffix}.wav', always_2d=True, dtype='float32')
        x = torch.from_numpy(x.T.copy())
        from .utils import convert_audio
        x = convert_audio(x, sr, model.sample_rate, model.channels)
        x = x[:, :model.sample_rate * 5]
        model.set_target_bandwidth(12)
        for use_lm in [False, True]:
            print(f"Doing {name}, use_lm={use_lm}")
            begin = time.time()
            res = compress(model, x, use_lm=use_lm)
            t_comp = time.time() - begin
            x_dec, _ = decompress(res)
            t_decomp = time.time() - begin - t_comp
            kbps = 8 * len(res) / 1000 / (x.shape[-1] / model.sample_rate)
            print(f"  kbps={kbps:.1f}  enc={t_comp:.2f}s  dec={t_decomp:.2f}s")
            assert x_dec.shape == x.shape


if __name__ == '__main__':
    test()
