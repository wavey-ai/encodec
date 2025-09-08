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
FP_SCALE = 1 << 11
MIN_RANGE = 6            # min bin width for arithmetic coder

def _counts_from_pdf(pdf: torch.Tensor, fp_scale: int) -> torch.Tensor:
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
        fx = torch.floor(x)
    return fx.to(torch.int64)

def _quantize_logits_(logits: torch.Tensor, step: float = LOGIT_QSTEP) -> torch.Tensor:
    y = (logits / step).to(torch.float64)
    eps = math.ldexp(1.0, -40)
    q = torch.floor(y + 0.5 - eps)
    return q * step

def _softmax_or_uniform(x: torch.Tensor, dim: int) -> torch.Tensor:
    s = _stable_softmax(x, dim)
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

def _stable_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    x = (logits - torch.amax(logits, dim=dim, keepdim=True)).to(torch.float64)
    z = torch.exp(x)
    z = z.movedim(dim, -1).contiguous()
    acc = z[..., 0].clone()
    for i in range(1, z.size(-1)):
        acc += z[..., i]
    out = (z / acc.unsqueeze(-1)).movedim(-1, dim)
    return out

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

    num = _counts_from_pdf(pdf, fp_scale).to(torch.int64)
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
    assert wav.dim() == 2
    if model.name not in MODELS:
        raise ValueError(f"Unsupported model {model.name}.")

    coder_device = torch.device("cpu")

    with torch.no_grad():
        frames = model.encode(wav[None])
    codes0, _ = frames[0]
    _, K, _ = codes0.shape

    lm = None
    if use_lm:
        lm = model.get_lm_model().to(dtype=torch.float64, device=coder_device).eval()

    metadata = {
        'm': model.name,
        'al': int(wav.shape[-1]),
        'nc': int(K),
        'lm': bool(use_lm),
        'fp': int(FP_SCALE),
        'acv': 3,
    }
    binary.write_ecdc_header(fo, metadata)

    for (frame, scale) in frames:
        if scale is not None:
            fo.write(struct.pack('!f', float(scale.cpu().item())))

        _B, _K, T = frame.shape
        if use_lm:
            coder = ArithmeticCoder(fo)
            states = None
            offset = 0
            input_ = torch.zeros(1, K, 1, dtype=torch.long, device=coder_device)
        else:
            packer = binary.BitPacker(model.bits_per_codebook, fo)

        for t in range(T):
            if use_lm:
                with torch.no_grad():
                    logits_raw, states, offset = lm.forward_logits(input_, states, offset)
                    logits_q = _quantize_logits_(logits_raw / lm.tau, LOGIT_QSTEP)
                    probas = _softmax_or_uniform(logits_q, dim=1)

                frame_slice = frame[:, :, t: t + 1].detach().to(coder_device)
                values = frame_slice[0, :, 0].tolist()
                for k, value in enumerate(values):
                    q_cdf = _deterministic_cdf(probas[0, :, k, 0], coder.total_range_bits, fp_scale=FP_SCALE, check=False)
                    coder.push(value, q_cdf)
                input_ = 1 + frame_slice
            else:
                values = frame[0, :, t].detach().cpu().tolist()
                for value in values:
                    packer.push(value)

        if use_lm:
            coder.flush()
        else:
            packer.flush()

def decompress_from_file(fo: tp.IO[bytes], device='cpu') -> tp.Tuple[torch.Tensor, int]:
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
    model_device = next(model.parameters()).device
    coder_device = torch.device("cpu")

    lm = None
    if use_lm:
        lm = model.get_lm_model().to(dtype=torch.float64, device=coder_device).eval()

    frames: tp.List[EncodedFrame] = []
    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length

    for offset_samples in range(0, audio_length, segment_stride):
        this_len = min(audio_length - offset_samples, segment_length)
        frame_length = int(math.ceil(this_len * model.frame_rate / model.sample_rate))

        if model.normalize:
            scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
            scale = torch.tensor(scale_f, device=coder_device).view(1)
        else:
            scale = None

        if use_lm:
            decoder = ArithmeticDecoder(fo)
            states = None
            offset = 0
            input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=coder_device)
        else:
            unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)

        frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=coder_device)

        for t in range(frame_length):
            if use_lm:
                with torch.no_grad():
                    logits_raw, states, offset = lm.forward_logits(input_, states, offset)
                    logits_q = _quantize_logits_(logits_raw / lm.tau, LOGIT_QSTEP)
                    probas = _softmax_or_uniform(logits_q, dim=1)

                code_list: tp.List[int] = []
                for k in range(num_codebooks):
                    q_cdf = _deterministic_cdf(probas[0, :, k, 0], decoder.total_range_bits, fp_scale=fp_scale, check=False)
                    code = decoder.pull(q_cdf)
                    if code is None:
                        raise EOFError("The stream ended sooner than expected.")
                    code_list.append(code)
                frame[0, :, t] = torch.tensor(code_list, dtype=torch.long, device=coder_device)
                input_ = 1 + frame[:, :, t: t + 1]
            else:
                code_list: tp.List[int] = []
                for _ in range(num_codebooks):
                    code = unpacker.pull()
                    if code is None:
                        raise EOFError("The stream ended sooner than expected.")
                    code_list.append(code)
                frame[0, :, t] = torch.tensor(code_list, dtype=torch.long, device=coder_device)

        frames.append((frame.to(model_device), None if scale is None else scale.to(model_device)))

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
