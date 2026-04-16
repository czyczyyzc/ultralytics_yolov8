#!/usr/bin/env python3
"""Export vendored NanoTrack to ONNX components aligned with RK3588 three-stage runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
VENDOR_ROOT = ROOT / "third_party" / "nanotrack_vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.utils.model_load import load_pretrain


class BackboneExportWrapper(nn.Module):
    """Export wrapper for template/search backbone plus neck."""

    def __init__(self, model: ModelBuilder):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model._extract(x)


class HeadExportWrapper(nn.Module):
    """Export wrapper for the NanoTrack head."""

    def __init__(self, model: ModelBuilder):
        super().__init__()
        self.head = model.ban_head

    def forward(self, template_feature, search_feature):
        cls, loc = self.head(template_feature, search_feature)
        return cls, loc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True, help="NanoTrack config yaml.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Training checkpoint or pretrained weights.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported ONNX files and manifest.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--template-size", type=int, default=127, help="Template input size.")
    parser.add_argument("--search-size", type=int, default=255, help="Search input size.")
    parser.add_argument("--manifest-name", default="export_manifest.json", help="Export summary filename.")
    parser.add_argument("--dry-run", action="store_true", help="Build the components and write a manifest without exporting ONNX.")
    return parser.parse_args()


def load_model(cfg_path: Path, snapshot_path: Path | None, device: torch.device) -> ModelBuilder:
    cfg.merge_from_file(str(cfg_path))
    cfg.CUDA = device.type == "cuda"
    model = ModelBuilder().to(device).eval()
    if snapshot_path:
        if not snapshot_path.exists():
            raise FileNotFoundError(f"NanoTrack snapshot not found: {snapshot_path}")
        load_pretrain(model, str(snapshot_path))
    return model


def export_onnx(module: nn.Module, inputs, output_path: Path, input_names: list[str], output_names: list[str], opset: int) -> None:
    torch.onnx.export(
        module,
        inputs,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def main() -> None:
    args = parse_args()
    cfg_path = args.cfg.expanduser().resolve()
    snapshot_path = args.snapshot.expanduser().resolve() if args.snapshot else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if not cfg_path.exists():
        raise FileNotFoundError(f"NanoTrack config not found: {cfg_path}")

    model = load_model(cfg_path, snapshot_path, device)
    backbone_wrapper = BackboneExportWrapper(model).to(device).eval()
    head_wrapper = HeadExportWrapper(model).to(device).eval()

    with torch.no_grad():
        template = torch.randn(1, 3, args.template_size, args.template_size, device=device)
        search = torch.randn(1, 3, args.search_size, args.search_size, device=device)
        template_feature = backbone_wrapper(template)
        search_feature = backbone_wrapper(search)
        cls, loc = head_wrapper(template_feature, search_feature)

    manifest = {
        "cfg": str(cfg_path),
        "snapshot": str(snapshot_path) if snapshot_path else "",
        "device": str(device),
        "template_input_shape_nchw": tensor_shape(template),
        "search_input_shape_nchw": tensor_shape(search),
        "template_feature_shape_nchw": tensor_shape(template_feature),
        "search_feature_shape_nchw": tensor_shape(search_feature),
        "head_cls_shape_nchw": tensor_shape(cls),
        "head_loc_shape_nchw": tensor_shape(loc),
        "exports": {
            "t_backbone": str((output_dir / "nanotrack_t_backbone.onnx").resolve()),
            "x_backbone": str((output_dir / "nanotrack_x_backbone.onnx").resolve()),
            "head": str((output_dir / "nanotrack_head.onnx").resolve()),
        },
        "notes": {
            "runtime_alignment": "Matches the Try2ChangeX RK3588 split runtime: template backbone, search backbone, head.",
            "layout": "Exported ONNX inputs and outputs use NCHW. RKNN conversion/runtime may transpose to NHWC as needed.",
        },
    }

    if not args.dry_run:
        if not torch.onnx.__dict__.get("export"):
            raise RuntimeError("PyTorch ONNX export is unavailable in this environment.")
        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("onnx is required for export. Install project optional dependency set 'export'.") from exc

        export_onnx(
            backbone_wrapper,
            (template,),
            output_dir / "nanotrack_t_backbone.onnx",
            input_names=["template"],
            output_names=["template_feature"],
            opset=args.opset,
        )
        export_onnx(
            backbone_wrapper,
            (search,),
            output_dir / "nanotrack_x_backbone.onnx",
            input_names=["search"],
            output_names=["search_feature"],
            opset=args.opset,
        )
        export_onnx(
            head_wrapper,
            (template_feature, search_feature),
            output_dir / "nanotrack_head.onnx",
            input_names=["template_feature", "search_feature"],
            output_names=["cls", "loc"],
            opset=args.opset,
        )

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
