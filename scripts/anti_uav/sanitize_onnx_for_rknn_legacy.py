#!/usr/bin/env python3
"""Sanitize an ONNX graph for older RKNN Toolkit2 releases."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import AttributeProto, helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ONNX model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX model path.")
    return parser.parse_args()


def strip_maxpool_dilations(model: onnx.ModelProto) -> int:
    changed = 0
    for node in model.graph.node:
        if node.op_type != "MaxPool":
            continue

        kept: list[AttributeProto] = []
        removed = False
        for attr in node.attribute:
            if attr.name == "dilations":
                values = helper.get_attribute_value(attr)
                if list(values) != [1, 1]:
                    raise ValueError(
                        f"Unexpected MaxPool dilations for node {node.name or node.output[0]}: {values}"
                    )
                removed = True
                continue
            kept.append(attr)

        if removed:
            changed += 1
            del node.attribute[:]
            node.attribute.extend(kept)
    return changed


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    model = onnx.load(str(input_path))
    changed = strip_maxpool_dilations(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"Saved sanitized ONNX to: {output_path}")
    print(f"Stripped MaxPool dilations attributes: {changed}")


if __name__ == "__main__":
    main()
