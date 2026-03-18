#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Decode EnCodec payloads across devices and compare corrupted pairs.")
    parser.add_argument("--payload-dir", type=Path, required=True, help="Directory containing .ecdc payload files.")
    parser.add_argument("--devices", nargs="+", default=["cpu"], help="Decode devices to test, e.g. cpu cuda.")
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("CLEAN", "CORRUPT"),
        default=[],
        help="Optional clean/corrupt filename pair to compare after decode.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def decode_payload(decompress, payload: bytes, device: str):
    wav, sr = decompress(payload, device=device)
    wav = wav.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav, sr


def compare_wavs(clean_wav: torch.Tensor, bad_wav: torch.Tensor, sr: int):
    n = min(clean_wav.shape[-1], bad_wav.shape[-1])
    clean_wav = clean_wav[..., :n]
    bad_wav = bad_wav[..., :n]
    diff = (bad_wav - clean_wav).abs()
    err = diff.amax(dim=0)
    mask = err > 1e-3
    first_bad = int(torch.argmax(mask.to(torch.int64)).item()) if bool(mask.any()) else None
    last_bad = int((mask.numel() - 1) - torch.argmax(mask.flip(0).to(torch.int64)).item()) if bool(mask.any()) else None
    return {
        "corruption_mae": float(diff.mean().item()),
        "corruption_max_abs": float(diff.max().item()),
        "first_bad_sample": first_bad,
        "last_bad_sample": last_bad,
        "bad_duration_s": None if first_bad is None else (last_bad - first_bad + 1) / sr,
    }


def main():
    args = parse_args()

    from encodec.compress import decompress

    payload_dir = args.payload_dir
    results = []
    pair_map = {tuple(pair) for pair in args.pair}

    for payload_path in sorted(payload_dir.glob("*.ecdc")):
        payload = payload_path.read_bytes()
        for device in args.devices:
            row = {"file": payload_path.name, "device": device}
            try:
                wav, sr = decode_payload(decompress, payload, device)
                row.update({
                    "success": True,
                    "sr": sr,
                    "shape": list(wav.shape),
                    "dtype": str(wav.dtype),
                    "max_abs": float(wav.abs().max().item()),
                })
            except Exception as exc:
                row.update({"success": False, "error": repr(exc)})
            results.append(row)

    for clean_name, corrupt_name in sorted(pair_map):
        clean_payload = payload_dir.joinpath(clean_name).read_bytes()
        corrupt_payload = payload_dir.joinpath(corrupt_name).read_bytes()
        for device in args.devices:
            row = {"clean": clean_name, "corrupt": corrupt_name, "device": device}
            try:
                clean_wav, sr = decode_payload(decompress, clean_payload, device)
                corrupt_wav, corrupt_sr = decode_payload(decompress, corrupt_payload, device)
                if sr != corrupt_sr:
                    raise RuntimeError(f"Sample rate mismatch: {sr} != {corrupt_sr}")
                row.update({"success": True, "sr": sr})
                row.update(compare_wavs(clean_wav, corrupt_wav, sr))
            except Exception as exc:
                row.update({"success": False, "error": repr(exc)})
            results.append(row)

    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
