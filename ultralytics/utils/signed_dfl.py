"""P3-only DFL bin geometry, in feature-grid units."""

import torch

MIN_BIN, MAX_BIN = -1, 15


def signed_dfl_logits(nonnegative, negative):
    """Assemble (..., 4, 17) from the legacy channels and four new -1 logits."""
    return torch.cat((negative.unsqueeze(-1), nonnegative.unflatten(-1, (4, MAX_BIN + 1))), dim=-1)


def signed_dfl_decode(logits):
    bins = torch.arange(MIN_BIN, MAX_BIN + 1, device=logits.device, dtype=torch.float32)
    return (logits.softmax(-1, dtype=torch.float32) * bins).sum(-1)


def ltrb(points, boxes):
    return torch.cat((points - boxes[..., :2], boxes[..., 2:] - points), dim=-1)


def representable(ltrb_targets):
    return ((ltrb_targets >= MIN_BIN) & (ltrb_targets <= MAX_BIN)).all(-1)
