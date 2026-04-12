from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import typing as tp

import onnx
import torch
from torch import nn

from .model import EncodecModel


MODEL_FACTORIES: dict[str, tp.Callable[..., EncodecModel]] = {
    "encodec_24khz": EncodecModel.encodec_model_24khz,
    "encodec_48khz": EncodecModel.encodec_model_48khz,
}


@dataclass
class OnnxFrameBundleMetadata:
    schema_version: int
    model_name: str
    bandwidth_kbps: float
    sample_rate: int
    channels: int
    segment_samples: int
    segment_stride: int
    normalize: bool
    num_codebooks: int
    frame_length: int
    encode_model: str
    decode_model: str
    opset_version: int


class EncodeFrameWrapper(nn.Module):
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        codes, scale = self.model._encode_frame(x)
        if scale is None:
            scale = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return codes, scale


class DecodeFrameWrapper(nn.Module):
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.model = model

    def forward(self, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.model.normalize:
            return self.model._decode_frame((codes, scale))
        return self.model._decode_frame((codes, None))


def build_model(
    model_name: str,
    bandwidth_kbps: float,
    device: str = "cpu",
    repository: Path | None = None,
) -> EncodecModel:
    if model_name not in MODEL_FACTORIES:
        supported = ", ".join(sorted(MODEL_FACTORIES.keys()))
        raise ValueError(f"Unsupported model {model_name!r}. Use one of: {supported}.")

    model = MODEL_FACTORIES[model_name](repository=repository)
    model.set_target_bandwidth(float(bandwidth_kbps))
    return model.to(device).eval()


def export_frame_onnx_bundle(
    output_dir: str | Path,
    model_name: str = "encodec_48khz",
    bandwidth_kbps: float = 6.0,
    device: str = "cpu",
    repository: str | Path | None = None,
    opset_version: int = 18,
) -> OnnxFrameBundleMetadata:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repository_path = None if repository is None else Path(repository)

    model = build_model(model_name, bandwidth_kbps, device=device, repository=repository_path)
    segment_samples = int(model.segment_length or model.sample_rate)

    torch.manual_seed(0)
    dummy_audio = torch.randn(
        1,
        model.channels,
        segment_samples,
        device=device,
        dtype=torch.float32,
    ) * 0.01

    encoder = EncodeFrameWrapper(model).eval()
    decoder = DecodeFrameWrapper(model).eval()

    with torch.no_grad():
        codes, scale = encoder(dummy_audio)
        codes = codes.detach().clone()
        scale = scale.detach().clone()

    encode_path = output_dir / "encode_frame.onnx"
    decode_path = output_dir / "decode_frame.onnx"

    torch.onnx.export(
        encoder,
        (dummy_audio,),
        encode_path,
        input_names=["audio"],
        output_names=["codes", "scale"],
        opset_version=opset_version,
        dynamo=False,
    )
    torch.onnx.export(
        decoder,
        (codes, scale),
        decode_path,
        input_names=["codes", "scale"],
        output_names=["audio"],
        opset_version=opset_version,
        dynamo=False,
    )

    onnx.checker.check_model(str(encode_path))
    onnx.checker.check_model(str(decode_path))

    metadata = OnnxFrameBundleMetadata(
        schema_version=1,
        model_name=model.name,
        bandwidth_kbps=float(bandwidth_kbps),
        sample_rate=int(model.sample_rate),
        channels=int(model.channels),
        segment_samples=segment_samples,
        segment_stride=int(model.segment_stride or segment_samples),
        normalize=bool(model.normalize),
        num_codebooks=int(codes.shape[1]),
        frame_length=int(codes.shape[2]),
        encode_model=encode_path.name,
        decode_model=decode_path.name,
        opset_version=int(opset_version),
    )
    (output_dir / "bundle.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n")
    return metadata


def metadata_to_json(metadata: OnnxFrameBundleMetadata) -> str:
    return json.dumps(asdict(metadata), indent=2, sort_keys=True)
