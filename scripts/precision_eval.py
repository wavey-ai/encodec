#!/usr/bin/env python3
import argparse
import io
import json
import math
import struct
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run EnCodec precision and robustness experiments.")
    parser.add_argument("--repo-path", type=Path, required=True, help="Path to the EnCodec checkout to evaluate.")
    parser.add_argument("--input", type=Path, required=True, help="Input audio file.")
    parser.add_argument("--model", choices=["encodec_24khz", "encodec_48khz"], default="encodec_48khz")
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--device", default="cpu", help="Encoding device, e.g. cpu or mps.")
    parser.add_argument("--decode-device", default=None, help="Decode device. Defaults to --device.")
    parser.add_argument("--lm", action="store_true", help="Enable LM entropy coding.")
    parser.add_argument("--segment", type=float, default=None, help="Model segment length in seconds.")
    parser.add_argument("--overlap", type=float, default=None, help="Model overlap fraction.")
    parser.add_argument("--offset", type=float, default=0.0, help="Clip start offset in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Clip duration in seconds.")
    parser.add_argument("--corrupt-byte-fraction", type=float, default=None, help="Flip one byte near this fraction of the payload.")
    parser.add_argument("--corrupt-byte-index", type=int, default=None, help="Flip one byte at this absolute payload index.")
    parser.add_argument("--output-payload", type=Path, default=None, help="Optional path to write the encoded payload.")
    parser.add_argument("--output-corrupt-payload", type=Path, default=None, help="Optional path to write the corrupted payload.")
    return parser.parse_args()


def load_audio(path: Path):
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = torch.from_numpy(wav.T.copy())
    return wav, sr


def clip_audio(wav: torch.Tensor, sr: int, offset_s: float, duration_s: float | None):
    start = max(0, int(round(offset_s * sr)))
    end = wav.shape[-1] if duration_s is None else min(wav.shape[-1], start + int(round(duration_s * sr)))
    return wav[:, start:end]


def flip_payload_byte(payload: bytes, metadata_len: int, byte_index: int):
    data = bytearray(payload)
    target = metadata_len + byte_index
    if target < metadata_len or target >= len(data):
        raise ValueError(f"Corruption index {byte_index} is out of range for payload of {len(data) - metadata_len} bytes.")
    data[target] ^= 0x01
    return bytes(data), target


def flip_chunk_body_byte(payload: bytes, metadata_len: int, metadata: dict, byte_index: int | None, fraction: float | None):
    chunk_header = struct.Struct("!II")
    data = bytearray(payload)
    stream = io.BytesIO(payload)
    stream.seek(metadata_len)

    body_ranges = []
    while stream.tell() < len(payload):
        header_pos = stream.tell()
        header = stream.read(chunk_header.size)
        if len(header) != chunk_header.size:
            break
        chunk_len, _chunk_crc = chunk_header.unpack(header)
        body_start = stream.tell()
        body_end = body_start + chunk_len
        if body_end > len(payload):
            break
        body_ranges.append((body_start, body_end, header_pos))
        stream.seek(body_end)

    if not body_ranges:
        raise ValueError("No chunk bodies found in payload.")

    total_body_bytes = sum(end - start for start, end, _ in body_ranges)
    if byte_index is not None:
        remaining = byte_index
    else:
        assert fraction is not None
        remaining = min(total_body_bytes - 1, max(0, int(math.floor(total_body_bytes * fraction))))

    chunk_index = 0
    target = None
    for idx, (start, end, _header_pos) in enumerate(body_ranges):
        chunk_len = end - start
        if remaining < chunk_len:
            target = start + remaining
            chunk_index = idx
            break
        remaining -= chunk_len

    if target is None or target >= len(data):
        raise ValueError("Corruption index is out of range for chunk bodies.")

    data[target] ^= 0x01
    return bytes(data), target, chunk_index, target - body_ranges[chunk_index][0]


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo_path))

    import encodec.binary as binary
    from encodec.compress import compress, decompress, MODELS
    from encodec.utils import convert_audio

    decode_device = args.decode_device or args.device
    wav, sr = load_audio(args.input)
    wav = clip_audio(wav, sr, args.offset, args.duration)
    source_duration = wav.shape[-1] / sr

    model = MODELS[args.model]().to(args.device)
    model.set_target_bandwidth(args.bandwidth)
    if args.segment is not None:
        model.segment = args.segment
    if args.overlap is not None:
        model.overlap = args.overlap

    wav_in = convert_audio(wav, sr, model.sample_rate, model.channels).to(args.device)
    wav_ref = wav_in.detach().cpu()

    t0 = time.perf_counter()
    clean_payload = compress(model, wav_in, use_lm=args.lm)
    encode_s = time.perf_counter() - t0

    if args.output_payload is not None:
        args.output_payload.parent.mkdir(parents=True, exist_ok=True)
        args.output_payload.write_bytes(clean_payload)

    payload = clean_payload
    header_stream = io.BytesIO(clean_payload)
    metadata = binary.read_ecdc_header(header_stream)
    payload_offset = header_stream.tell()

    corrupt_abs = None
    corrupt_chunk_index = None
    corrupt_chunk_byte = None
    if args.corrupt_byte_index is not None:
        if metadata.get("acv") == 4:
            payload, corrupt_abs, corrupt_chunk_index, corrupt_chunk_byte = flip_chunk_body_byte(
                payload, payload_offset, metadata, args.corrupt_byte_index, None)
        else:
            payload, corrupt_abs = flip_payload_byte(payload, payload_offset, args.corrupt_byte_index)
    elif args.corrupt_byte_fraction is not None:
        if metadata.get("acv") == 4:
            payload, corrupt_abs, corrupt_chunk_index, corrupt_chunk_byte = flip_chunk_body_byte(
                payload, payload_offset, metadata, None, args.corrupt_byte_fraction)
        else:
            data_len = len(clean_payload) - payload_offset
            corrupt_idx = min(data_len - 1, max(0, int(math.floor(data_len * args.corrupt_byte_fraction))))
            payload, corrupt_abs = flip_payload_byte(payload, payload_offset, corrupt_idx)

    if args.output_corrupt_payload is not None:
        args.output_corrupt_payload.parent.mkdir(parents=True, exist_ok=True)
        args.output_corrupt_payload.write_bytes(payload)

    result = {
        "repo_path": str(args.repo_path),
        "input": str(args.input),
        "model": args.model,
        "bandwidth": args.bandwidth,
        "device": args.device,
        "decode_device": decode_device,
        "lm": args.lm,
        "segment": model.segment,
        "overlap": model.overlap,
        "input_sr": sr,
        "model_sr": model.sample_rate,
        "input_channels": int(wav.shape[0]),
        "model_channels": int(model.channels),
        "source_duration_s": source_duration,
        "encoded_samples": int(wav_in.shape[-1]),
        "encoded_bytes": len(clean_payload),
        "payload_bytes": len(clean_payload) - payload_offset,
        "output_payload": None if args.output_payload is None else str(args.output_payload),
        "output_corrupt_payload": None if args.output_corrupt_payload is None else str(args.output_corrupt_payload),
        "header_metadata": metadata,
        "corrupt_absolute_byte": corrupt_abs,
        "corrupt_payload_byte": None if corrupt_abs is None else corrupt_abs - payload_offset,
        "corrupt_chunk_index": corrupt_chunk_index,
        "corrupt_chunk_byte": corrupt_chunk_byte,
    }

    try:
        clean_decode = None
        if payload != clean_payload:
            clean_decode, _ = decompress(clean_payload, device=decode_device)
            clean_decode = clean_decode.detach().cpu()
            if clean_decode.dim() == 1:
                clean_decode = clean_decode.unsqueeze(0)

        t1 = time.perf_counter()
        wav_out, out_sr = decompress(payload, device=decode_device)
        decode_s = time.perf_counter() - t1
        wav_out = wav_out.detach().cpu()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        wav_out = wav_out[:, :wav_ref.shape[-1]]
        if wav_out.shape[-1] < wav_ref.shape[-1]:
            pad = wav_ref.shape[-1] - wav_out.shape[-1]
            wav_out = torch.nn.functional.pad(wav_out, (0, pad))
        diff = wav_out - wav_ref
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
        signal_power = float(wav_ref.pow(2).mean().item())
        snr_db = float("inf") if mse == 0 else 10.0 * math.log10(max(signal_power, 1e-12) / mse)
        result.update({
            "success": True,
            "decode_sr": out_sr,
            "decoded_samples": int(wav_out.shape[-1]),
            "encode_s": encode_s,
            "decode_s": decode_s,
            "rtf_encode": encode_s / max(source_duration, 1e-9),
            "rtf_decode": decode_s / max(source_duration, 1e-9),
            "mse": mse,
            "mae": mae,
            "max_abs_err": float(diff.abs().max().item()),
            "snr_db": snr_db,
            "bps": (len(payload) * 8.0) / max(source_duration, 1e-9),
        })
        if clean_decode is not None:
            clean_cmp = clean_decode[:, :wav_out.shape[-1]]
            if clean_cmp.shape[-1] < wav_out.shape[-1]:
                clean_cmp = torch.nn.functional.pad(clean_cmp, (0, wav_out.shape[-1] - clean_cmp.shape[-1]))
            corr_diff = wav_out - clean_cmp
            err = corr_diff.abs().amax(dim=0)
            mask = err > 1e-3
            first_bad = int(torch.argmax(mask.to(torch.int64)).item()) if bool(mask.any()) else None
            last_bad = int((mask.numel() - 1) - torch.argmax(mask.flip(0).to(torch.int64)).item()) if bool(mask.any()) else None
            result.update({
                "corruption_mae_vs_clean_decode": float(corr_diff.abs().mean().item()),
                "corruption_max_abs_vs_clean_decode": float(corr_diff.abs().max().item()),
                "corruption_first_bad_sample": first_bad,
                "corruption_last_bad_sample": last_bad,
                "corruption_bad_duration_s": None if first_bad is None else (last_bad - first_bad + 1) / out_sr,
            })
    except Exception as exc:
        result.update({
            "success": False,
            "encode_s": encode_s,
            "decode_error": repr(exc),
            "bps": (len(payload) * 8.0) / max(source_duration, 1e-9),
        })

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
