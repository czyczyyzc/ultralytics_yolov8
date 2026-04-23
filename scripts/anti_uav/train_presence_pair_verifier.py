#!/usr/bin/env python3
"""Train a small ROI verifier head over template/current anti-UAV patches."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, nargs="+", required=True, help="One or more exported pair manifests.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint.")
    parser.add_argument("--history-json", type=Path, default=None, help="Optional training history JSON.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the pair head.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--loss-mode", choices=("ce", "edl"), default="edl", help="Classification loss.")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata features and train image-only.")
    parser.add_argument("--edl-annealing-epochs", type=int, default=6, help="Epochs for EDL KL annealing.")
    return parser.parse_args()


def parse_torch_device(torch_module, device: str):
    """Resolve a single torch device string."""
    value = str(device).strip()
    if not value:
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    if "," in value:
        value = value.split(",", 1)[0].strip()
    if value.isdigit():
        value = f"cuda:{value}"
    return torch_module.device(value)


def load_manifest_rows(paths: list[Path]) -> list[dict]:
    """Load all manifest rows."""
    rows = []
    for path in paths:
        with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if raw:
                    rows.append(json.loads(raw))
    if not rows:
        raise ValueError("No training rows found.")
    return rows


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate train/val indices while preserving both classes when possible."""
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(indices)
        val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = max(1, min(len(shuffled) - 1, val_count))
        else:
            val_count = 0
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def load_pair_image(template_path: str, search_path: str) -> np.ndarray:
    """Load two grayscale crops and stack them into a 2-channel input."""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if template is None or search is None:
        raise FileNotFoundError(f"Unable to load pair crops: {template_path}, {search_path}")
    template = template.astype(np.float32) / 255.0
    search = search.astype(np.float32) / 255.0
    return np.stack([template, search], axis=0)


def build_arrays(rows: list[dict], use_metadata: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert manifest rows into numpy arrays."""
    feature_names = tuple(solutions.HeuristicPresenceVerifier.feature_names)
    images, features, labels = [], [], []
    for row in rows:
        images.append(load_pair_image(row["template_path"], row["search_path"]))
        features.append([float(row["features"].get(name, 0.0)) for name in feature_names] if use_metadata else [])
        labels.append(int(row["label"]))
    image_array = np.asarray(images, dtype=np.float32)
    feature_array = np.asarray(features, dtype=np.float32) if use_metadata else np.zeros((len(images), 0), dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    return image_array, feature_array, label_array


def dirichlet_kl(alpha, num_classes: int, torch_module):
    """KL divergence between a Dirichlet and the uniform prior."""
    beta = torch_module.ones((1, num_classes), device=alpha.device, dtype=alpha.dtype)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    ln_b_alpha = torch_module.lgamma(sum_alpha) - torch_module.lgamma(alpha).sum(dim=1, keepdim=True)
    ln_b_beta = torch_module.lgamma(beta).sum(dim=1, keepdim=True) - torch_module.lgamma(sum_beta)
    digamma_diff = torch_module.digamma(alpha) - torch_module.digamma(sum_alpha)
    return ((alpha - beta) * digamma_diff).sum(dim=1, keepdim=True) + ln_b_alpha + ln_b_beta


def edl_mse_loss(logits, labels, epoch: int, annealing_epochs: int, torch_module):
    """Evidential deep learning classification loss."""
    num_classes = logits.shape[1]
    evidence = torch_module.nn.functional.softplus(logits)
    alpha = evidence + 1.0
    target = torch_module.nn.functional.one_hot(labels, num_classes=num_classes).float()
    total_evidence = alpha.sum(dim=1, keepdim=True)
    prediction = alpha / total_evidence
    mse = ((target - prediction) ** 2).sum(dim=1, keepdim=True)
    var = (alpha * (total_evidence - alpha) / (total_evidence * total_evidence * (total_evidence + 1.0))).sum(
        dim=1, keepdim=True
    )
    alpha_tilde = (alpha - 1.0) * (1.0 - target) + 1.0
    anneal = min(1.0, float(epoch + 1) / max(float(annealing_epochs), 1.0))
    loss = mse + var + anneal * dirichlet_kl(alpha_tilde, num_classes, torch_module)
    return loss.mean()


def evaluate(model, image_array, feature_array, labels, device, torch_module, loss_mode: str, annealing_epochs: int, epoch: int):
    """Evaluate the verifier and return common metrics."""
    if len(labels) == 0:
        return {"loss": None, "accuracy": None, "precision": None, "recall": None, "f1": None, "uncertainty": None}

    criterion = torch_module.nn.CrossEntropyLoss()
    image_tensor = torch_module.from_numpy(image_array).to(device)
    feature_tensor = torch_module.from_numpy(feature_array).to(device) if feature_array.shape[1] > 0 else None
    label_tensor = torch_module.from_numpy(labels).to(device)
    model.eval()
    with torch_module.no_grad():
        logits = model(image_tensor, feature_tensor)
        if loss_mode == "edl":
            loss = edl_mse_loss(logits, label_tensor, epoch, annealing_epochs, torch_module).item()
            evidence = torch_module.nn.functional.softplus(logits)
            alpha = evidence + 1.0
            total_evidence = alpha.sum(dim=1, keepdim=True)
            probs = (alpha / total_evidence)[:, 1].cpu().numpy()
            uncertainty = (2.0 / total_evidence).clamp(max=1.0).mean().item()
        else:
            loss = criterion(logits, label_tensor).item()
            probs = torch_module.softmax(logits, dim=1)[:, 1].cpu().numpy()
            uncertainty = np.mean(1.0 - np.abs(probs - 0.5) * 2.0)
    preds = (probs >= 0.5).astype(np.int64)
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
        "uncertainty": float(uncertainty),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch = __import__("torch")
    device = parse_torch_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    rows = load_manifest_rows(args.manifest_jsonl)
    use_metadata = not args.no_metadata
    feature_names = tuple(solutions.HeuristicPresenceVerifier.feature_names)
    images, features, labels = build_arrays(rows, use_metadata=use_metadata)
    train_indices, val_indices = stratified_split(labels, args.val_ratio, args.seed)
    train_images, train_features, train_labels = images[train_indices], features[train_indices], labels[train_indices]
    val_images, val_features, val_labels = images[val_indices], features[val_indices], labels[val_indices]

    metadata_dim = train_features.shape[1] if use_metadata else 0
    model = solutions.PairPresenceNet(in_channels=2, metadata_dim=metadata_dim, hidden_dim=args.hidden_dim).model.to(device)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_features) if use_metadata else torch.zeros((len(train_images), 0), dtype=torch.float32),
        torch.from_numpy(train_labels),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)

    history = []
    best_state = None
    best_metric = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_images, batch_features, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_features = batch_features.to(device) if use_metadata else None
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images, batch_features)
            if args.loss_mode == "edl":
                loss = edl_mse_loss(logits, batch_labels, epoch, args.edl_annealing_epochs, torch)
            else:
                loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_images.size(0)

        train_loss = running_loss / max(len(train_labels), 1)
        metrics = evaluate(
            model,
            val_images,
            val_features,
            val_labels,
            device,
            torch,
            args.loss_mode,
            args.edl_annealing_epochs,
            epoch,
        )
        entry = {"epoch": epoch + 1, "train_loss": float(train_loss), **metrics}
        history.append(entry)
        metric = (metrics.get("f1") or 0.0) - 0.25 * (metrics.get("uncertainty") or 0.0)
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    payload = {
        "state_dict": best_state or model.state_dict(),
        "feature_names": feature_names,
        "hidden_dim": int(args.hidden_dim),
        "patch_size": int(images.shape[-1]),
        "loss_mode": args.loss_mode,
        "use_metadata": use_metadata,
        "history": history,
        "train_samples": int(len(train_labels)),
        "val_samples": int(len(val_labels)),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.output))

    if args.history_json is not None:
        args.history_json = args.history_json.expanduser().resolve()
        args.history_json.parent.mkdir(parents=True, exist_ok=True)
        args.history_json.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "output": str(args.output),
        "best_metric": float(best_metric),
        "train_samples": int(len(train_labels)),
        "val_samples": int(len(val_labels)),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
