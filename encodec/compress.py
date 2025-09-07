# encodec/compress.py
# Deterministic coder: architecture-stable CDF construction + logit quantization.

import io
import math
import struct
import typing as tp

import torch

from . import binary
from .model import EncodecModel, EncodedFrame
from .quantization.ac import (
    ArithmeticCoder,
    ArithmeticDecoder,
)

# Hard determinism toggles
torch.use_deterministic_algorithms(True)
torch.backends.mkldnn.enabled = False

# Registry
MODELS = {
    'encodec_24khz': EncodecModel.encodec_model_24khz,
    'encodec_48khz': EncodecModel.encodec_model_48khz,
}

# Chosen scales for stability vs. compression efficiency
# - LOGIT_QSTEP: coarse enough to suppress tiny arch drift, fine enough to preserve coding gain
# - FP_SCALE:    count scale used before integer range allocation inside the CDF
LOGIT_QSTEP = 1.0 / 64.0
FP_SCALE = 1 << 14       # 16384; lower than 1<<16 for better cross-arch stability
ROUND_CDF = 1e-4         # unused in this deterministic path, kept for signature parity
MIN_RANGE = 2            # min bin width for arithmetic coder


def _quantize_logits_(logits: torch.Tensor, step: float = LOGIT_QSTEP) -> torch.Tensor:
    # In-place-ish quantization without breaking autograd (we're in no_grad anyway).
    return torch.round(logits / step) * step


def _stable_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    # f64 softmax with explicit max subtraction for numerical stability
    m = torch.amax(logits, dim=dim, keepdim=True)
    z = torch.exp((logits - m).to(torch.float64))
    s = torch.sum(z, dim=dim, keepdim=True)
    # safeguard in case of weird NaNs/Inf
    bad = ~torch.isfinite(s) | (s <= 0)
    if bad.any():
        # replace by uniform
        z = torch.ones_like(z, dtype=torch.float64)
        s = torch.sum(z, dim=dim, keepdim=True)
    return z / s


def _deterministic_cdf(pdf: torch.Tensor,
                       total_range_bits: int,
                       fp_scale: int = FP_SCALE,
                       min_range: int = MIN_RANGE,
                       check: bool = False) -> torch.Tensor:
    """
    Architecture-stable integer CDF:
      1) clamp pdf; compute integer "counts" by floor(pdf * fp_scale) in f64
      2) allocate the remaining counts deterministically by priority
      3) add min_range to each bin, cum-sum to final CDF that sums to 2^bits
    Any tiny floating diffs that don't change floor() outputs produce identical CDFs.
    """
    pdf = pdf.detach().to(torch.float64).clamp_min(0)
    s = pdf.sum()
    if (not torch.isfinite(s)) or (s <= 0):
        pdf = torch.ones_like(pdf)
        s = pdf.sum()

    num = torch.floor(pdf * fp_scale).to(torch.int64)
    if int(num.sum().item()) <= 0:
        num = torch.ones_like(num)

    total = 1 << total_range_bits
    n = int(num.numel())
    alloc = total - min_range * n
    num_sum = int(num.sum().item())

    # base integer allocation
    base = (alloc * num) // num_sum
    remainder = int(alloc - int(base.sum().item()))
    if remainder > 0:
        # deterministic priority: residual * (n+1) - index  (stable sort)
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


def compress_to_file(model: EncodecModel, wav: torch.Tensor, fo: tp.IO[bytes],
                     use_lm: bool = True):
    """
    Compress a waveform to a file-object using the given model.
    Deterministic path is enforced unconditionally (no metadata flags).
    """
    assert wav.dim() == 2, "Expected [C, T]."
    if model.name not in MODELS:
        raise ValueError(f"Unsupported model {model.name}.")

    device = wav.device

    # Encode once to know frames and K
    with torch.no_grad():
        frames = model.encode(wav[None])
    codes0, _ = frames[0]
    _, K, _ = codes0.shape

    # Language model (float64), but logits quantized before softmax
    lm = None
    if use_lm:
        lm = model.get_lm_model().to(dtype=torch.float64, device=device)
        lm.eval()

    # Minimal, unchanged metadata set (no new flags)
    metadata = {
        'm': model.name,
        'al': int(wav.shape[-1]),
        'nc': int(K),
        'lm': bool(use_lm),
        'fp': int(FP_SCALE),
        'acv': 3,
    }
    binary.write_ecdc_header(fo, metadata)

    # Bitstream
    for (frame, scale) in frames:
        if scale is not None:
            fo.write(struct.pack('!f', float(scale.cpu().item())))

        _B, _K, T = frame.shape
        if use_lm:
            coder = ArithmeticCoder(fo)
            states: tp.Any = None
            offset = 0
            input_ = torch.zeros(1, K, 1, dtype=torch.long, device=device)
        else:
            packer = binary.BitPacker(model.bits_per_codebook, fo)

        for t in range(T):
            if use_lm:
                with torch.no_grad():
                    probas_raw, states, offset = lm(input_, states, offset)   # [1, card, K, 1]
                    # Quantize logits (rebuild from probs with log if needed)
                    # Safer: pass-through by reverse softmax → logits, quantize, softmax
                    # But we only get probas. So enforce quantization by re-logit with clip.
                    # To avoid log(0), clamp and then re-softmax.
                    p = torch.clamp(probas_raw, min=1e-12)
                    logits = torch.log(p)
                    logits_q = _quantize_logits_(logits, LOGIT_QSTEP)
                    probas = _stable_softmax(logits_q, dim=1)
            for k, value in enumerate(frame[0, :, t].tolist()):
                if use_lm:
                    q_cdf = _deterministic_cdf(probas[0, :, k, 0], coder.total_range_bits, fp_scale=FP_SCALE, check=False)
                    coder.push(value, q_cdf)
                else:
                    packer.push(value)
            if use_lm:
                input_ = 1 + frame[:, :, t: t + 1]

        if use_lm:
            coder.flush()
        else:
            packer.flush()


def decompress_from_file(fo: tp.IO[bytes], device='cpu') -> tp.Tuple[torch.Tensor, int]:
    """
    Decompress from a file-object. Deterministic path (matching encoder) is used unconditionally.
    """
    metadata = binary.read_ecdc_header(fo)
    model_name = metadata['m']
    audio_length = int(metadata['al'])
    num_codebooks = int(metadata['nc'])
    use_lm = bool(metadata['lm'])
    fp_scale = int(metadata.get('fp', FP_SCALE))
    acv = int(metadata.get('acv', 0))

    if model_name not in MODELS:
        raise ValueError(f"Unsupported model {model_name}.")
    if acv != 3:
        raise ValueError("Unsupported bitstream version; re-encode with this coder.")

    model = MODELS[model_name]().to(device)

    lm = None
    if use_lm:
        lm = model.get_lm_model().to(dtype=torch.float64, device=device)
        lm.eval()

    frames: tp.List[EncodedFrame] = []
    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length

    for offset_samples in range(0, audio_length, segment_stride):
        this_len = min(audio_length - offset_samples, segment_length)
        frame_length = int(math.ceil(this_len * model.frame_rate / model.sample_rate))

        if model.normalize:
            scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
            scale = torch.tensor(scale_f, device=device).view(1)
        else:
            scale = None

        if use_lm:
            decoder = ArithmeticDecoder(fo)
            states: tp.Any = None
            offset = 0
            input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=device)
        else:
            unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)

        frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=device)

        for t in range(frame_length):
            if use_lm:
                with torch.no_grad():
                    probas_raw, states, offset = lm(input_, states, offset)
                    p = torch.clamp(probas_raw, min=1e-12)
                    logits = torch.log(p)
                    logits_q = _quantize_logits_(logits, LOGIT_QSTEP)
                    probas = _stable_softmax(logits_q, dim=1)

            code_list: tp.List[int] = []
            for k in range(num_codebooks):
                if use_lm:
                    q_cdf = _deterministic_cdf(probas[0, :, k, 0], decoder.total_range_bits, fp_scale=fp_scale, check=False)
                    code = decoder.pull(q_cdf)
                else:
                    code = unpacker.pull()
                if code is None:
                    raise EOFError("The stream ended sooner than expected.")
                code_list.append(code)

            frame[0, :, t] = torch.tensor(code_list, dtype=torch.long, device=device)
            if use_lm:
                input_ = 1 + frame[:, :, t: t + 1]

        frames.append((frame, scale))

    with torch.no_grad():
        wav = model.decode(frames)
    return wav[0, :, :audio_length], model.sample_rate


def compress(model: EncodecModel, wav: torch.Tensor, use_lm: bool = False) -> bytes:
    fo = io.BytesIO()
    compress_to_file(model, wav, fo, use_lm=use_lm)
    return fo.getvalue()


def decompress(compressed: bytes, device='cpu') -> tp.Tuple[torch.Tensor, int]:
    fo = io.BytesIO(compressed)
    return decompress_from_file(fo, device=device)

