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
    bits_per_codebook: int
    codebook_cardinality: int
    encode_model: str
    decode_model: str
    opset_version: int
    lm_model: str | None = None
    lm_dim: int | None = None
    lm_num_layers: int | None = None
    lm_past_context: int | None = None
    lm_logit_step: float | None = None
    lm_cardinality: int | None = None
    lm_dtype: str | None = None


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


class LmLogitsWrapper(nn.Module):
    def __init__(self, lm_model: nn.Module):
        super().__init__()
        self.lm_model = lm_model
        self.num_state_tensors = len(self.lm_model.transformer.layers)

    def forward(
        self,
        indices: torch.Tensor,
        offset: torch.Tensor,
        *states: torch.Tensor,
    ) -> tp.Any:
        logits, new_states, next_offset = self.lm_model.forward_logits(
            indices,
            list(states),
            offset,
        )
        return logits, next_offset, *new_states


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
    export_lm: bool = False,
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
    lm_path = output_dir / "lm_logits.onnx"

    torch.onnx.export(
        encoder,
        (dummy_audio,),
        encode_path,
        input_names=["audio"],
        output_names=["codes", "scale"],
        opset_version=opset_version,
        dynamo=False,
        dynamic_axes={
            "audio": {0: "batch"},
            "codes": {0: "batch"},
            "scale": {0: "batch"},
        },
    )
    torch.onnx.export(
        decoder,
        (codes, scale),
        decode_path,
        input_names=["codes", "scale"],
        output_names=["audio"],
        opset_version=opset_version,
        dynamo=False,
        dynamic_axes={
            "codes": {0: "batch"},
            "scale": {0: "batch"},
            "audio": {0: "batch"},
        },
    )

    lm_model_name = None
    lm_dim = None
    lm_num_layers = None
    lm_past_context = None
    lm_logit_step = None
    lm_cardinality = None
    lm_dtype = None
    if export_lm:
        lm = model.get_lm_model(device=torch.device(device), dtype=torch.float32).eval()
        lm_wrapper = LmLogitsWrapper(lm).eval()
        lm_num_layers = len(lm.transformer.layers)
        lm_dim = int(lm.dim)
        lm_past_context = int(lm.transformer.past_context)
        lm_logit_step = float(lm.logit_step)
        lm_cardinality = int(lm.card)
        lm_dtype = "float32"
        active_codebooks = int(codes.shape[1])
        dummy_indices = torch.zeros(
            1,
            active_codebooks,
            1,
            dtype=torch.long,
            device=device,
        )
        dummy_offset = torch.zeros((), dtype=torch.long, device=device)
        dummy_states = tuple(
            torch.zeros(
                (1, lm.transformer.past_context, lm.dim),
                dtype=lm.dtype,
                device=device,
            )
            for _ in range(lm_num_layers)
        )
        input_names = ["indices", "offset", *[f"state_{index}" for index in range(lm_num_layers)]]
        output_names = ["logits", "offset_out", *[f"next_state_{index}" for index in range(lm_num_layers)]]
        dynamic_axes: dict[str, dict[int, str]] = {
            "indices": {0: "batch", 2: "steps"},
            "logits": {0: "batch", 3: "steps"},
        }
        for index in range(lm_num_layers):
            dynamic_axes[f"state_{index}"] = {0: "batch", 1: "context"}
            dynamic_axes[f"next_state_{index}"] = {0: "batch", 1: "context"}
        torch.onnx.export(
            lm_wrapper,
            (dummy_indices, dummy_offset, *dummy_states),
            lm_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes=dynamic_axes,
        )
        onnx.checker.check_model(str(lm_path))
        lm_model_name = lm_path.name

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
        bits_per_codebook=int(model.bits_per_codebook),
        codebook_cardinality=int(model.quantizer.bins),
        encode_model=encode_path.name,
        decode_model=decode_path.name,
        lm_model=lm_model_name,
        lm_dim=lm_dim,
        lm_num_layers=lm_num_layers,
        lm_past_context=lm_past_context,
        lm_logit_step=lm_logit_step,
        lm_cardinality=lm_cardinality,
        lm_dtype=lm_dtype,
        opset_version=int(opset_version),
    )
    (output_dir / "bundle.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n")
    return metadata


def metadata_to_json(metadata: OnnxFrameBundleMetadata) -> str:
    return json.dumps(asdict(metadata), indent=2, sort_keys=True)
