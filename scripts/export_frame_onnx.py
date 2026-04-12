#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from encodec.onnx import export_frame_onnx_bundle, metadata_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the EnCodec frame encoder/decoder boundary to an ONNX bundle."
    )
    parser.add_argument(
        "--model",
        default="encodec_48khz",
        choices=["encodec_24khz", "encodec_48khz"],
        help="Pretrained EnCodec model to export.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=6.0,
        help="Target bandwidth in kbps for the exported bundle.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive encode_frame.onnx, decode_frame.onnx, and bundle.json.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for export, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help="Optional local checkpoint repository path.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_frame_onnx_bundle(
        output_dir=args.output_dir,
        model_name=args.model,
        bandwidth_kbps=args.bandwidth,
        device=args.device,
        repository=args.repository,
        opset_version=args.opset_version,
    )
    print(metadata_to_json(metadata))


if __name__ == "__main__":
    main()
