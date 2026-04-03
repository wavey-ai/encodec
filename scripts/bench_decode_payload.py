#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark EnCodec payload decode.")
    parser.add_argument("--repo-path", type=Path, required=True, help="Path to the EnCodec checkout.")
    parser.add_argument("--payload", type=Path, required=True, help="Path to the .ecdc payload.")
    parser.add_argument("--device", default="cpu", help="Decode device.")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup decodes to discard.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of decode repetitions.")
    parser.add_argument("--output-wav", type=Path, default=None, help="Optional WAV output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo_path))

    from encodec.compress import decompress

    payload = args.payload.read_bytes()
    runs = []
    wav_sha256 = None
    wav_shape = None
    sample_rate = None

    for _ in range(max(0, int(args.warmup))):
        wav, sample_rate = decompress(payload, device=args.device)
        wav_cpu = wav.detach().cpu().contiguous()
        digest = hashlib.sha256(wav_cpu.numpy().tobytes()).hexdigest()
        if wav_sha256 is None:
            wav_sha256 = digest
            wav_shape = list(wav_cpu.shape)
        elif digest != wav_sha256:
            raise RuntimeError(
                f"Non-deterministic warmup decode: first hash {wav_sha256}, later hash {digest}."
            )

    for _ in range(max(1, int(args.repeats))):
        t0 = time.perf_counter()
        wav, sample_rate = decompress(payload, device=args.device)
        decode_s = time.perf_counter() - t0
        wav_cpu = wav.detach().cpu().contiguous()
        digest = hashlib.sha256(wav_cpu.numpy().tobytes()).hexdigest()
        if wav_sha256 is None:
            wav_sha256 = digest
            wav_shape = list(wav_cpu.shape)
        elif digest != wav_sha256:
            raise RuntimeError(
                f"Non-deterministic decode: first hash {wav_sha256}, later hash {digest}."
            )
        runs.append(decode_s)

    result = {
        "payload": str(args.payload),
        "device": args.device,
        "warmup": max(0, int(args.warmup)),
        "repeats": len(runs),
        "decode_s_runs": runs,
        "decode_s_mean": sum(runs) / len(runs),
        "wav_sha256": wav_sha256,
        "wav_shape": wav_shape,
        "sample_rate": sample_rate,
    }
    if args.output_wav is not None:
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(args.output_wav),
            wav.detach().cpu().transpose(0, 1).numpy(),
            int(sample_rate),
            subtype="PCM_16",
        )
        result["output_wav"] = str(args.output_wav)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
