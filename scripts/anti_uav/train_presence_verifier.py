#!/usr/bin/env python3
"""Train a tiny MLP presence verifier over Scheme A anti-UAV feature logs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, nargs="+", required=True, help="One or more presence dataset JSONL files.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint path.")
    parser.add_argument("--history-json", type=Path, default=None, help="Optional training history JSON path.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension of the tiny MLP.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for a stratified split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def load_samples(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load JSONL feature rows into numpy arrays."""
    feature_names = tuple(solutions.HeuristicPresenceVerifier.feature_names)
    features, labels = [], []
    for path in paths:
        with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                features.append([float(row["features"].get(name, 0.0)) for name in feature_names])
                labels.append(int(row["label"]))
    if not features:
        raise ValueError("No training samples found.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def parse_torch_device(torch_module, device: str):
    """Resolve a single torch device string without depending on anti_uav internals."""
    value = str(device).strip()
    if not value:
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    if "," in value:
        value = value.split(",", 1)[0].strip()
    if value.isdigit():
        value = f"cuda:{value}"
    return torch_module.device(value)


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate train/val indices while keeping both classes represented when possible."""
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in sorted(set(labels.tolist())):
        label_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(label_indices)
        val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = max(1, min(len(shuffled) - 1, val_count))
        else:
            val_count = 0
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def evaluate(model, features, labels, device, torch_module) -> dict:
    """Compute simple validation metrics."""
    if len(labels) == 0:
        return {"loss": None, "accuracy": None, "precision": None, "recall": None}

    criterion = torch_module.nn.CrossEntropyLoss()
    tensor_x = torch_module.from_numpy(features).to(device)
    tensor_y = torch_module.from_numpy(labels).to(device)
    model.eval()
    with torch_module.no_grad():
        logits = model(tensor_x)
        loss = criterion(logits, tensor_y).item()
        preds = logits.argmax(dim=1).cpu().numpy()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    accuracy = float((preds == labels).mean())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch = __import__("torch")
    feature_names = tuple(solutions.HeuristicPresenceVerifier.feature_names)
    device = parse_torch_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    features, labels = load_samples(args.dataset_jsonl)
    train_indices, val_indices = stratified_split(labels, args.val_ratio, args.seed)
    train_x, train_y = features[train_indices], labels[train_indices]
    val_x, val_y = features[val_indices], labels[val_indices]

    network = solutions.PresenceMLP(len(feature_names), hidden_dim=args.hidden_dim).model.to(device)
    class_counts = np.bincount(train_y, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)

    history = []
    best_state = None
    best_metric = float("-inf")
    for epoch in range(1, args.epochs + 1):
        network.train()
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / max(len(dataset), 1)
        val_metrics = evaluate(network, val_x, val_y, device, torch)
        entry = {"epoch": epoch, "train_loss": float(train_loss), **val_metrics}
        history.append(entry)
        metric = val_metrics.get("f1") if val_metrics.get("f1") is not None else -train_loss
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}

    payload = {
        "state_dict": best_state or network.state_dict(),
        "feature_names": feature_names,
        "hidden_dim": int(args.hidden_dim),
        "history": history,
        "train_samples": int(len(train_y)),
        "val_samples": int(len(val_y)),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.output))

    if args.history_json is not None:
        args.history_json = args.history_json.expanduser().resolve()
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        args.history_json.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"output": str(args.output), "best_metric": best_metric, "train_samples": len(train_y), "val_samples": len(val_y)}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
