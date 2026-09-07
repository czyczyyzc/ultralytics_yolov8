"""Previous-frame-only ROI policy; intentionally has no annotation dependency."""

from __future__ import annotations


def choose_region(previous_tracks, frame_index, width, height, zoom=2,
                  refresh_interval=10, previous_anchor=None):
    if zoom not in (1, 2, 4) or refresh_interval < 1:
        raise ValueError("zoom must be 1, 2 or 4 and refresh_interval positive")
    full = (0, 0, width, height)
    if zoom == 1:
        return full, "full_baseline", None
    # Only confirmed, observation-backed outputs from frame t-1 are eligible.
    eligible = [t for t in previous_tracks if t["confirmed"] and not t["predicted"]]
    if not eligible:
        return full, "full_search", None
    anchor = next((t for t in eligible if t["id"] == previous_anchor), None)
    if anchor is None:
        anchor = max(eligible, key=lambda t: (t["hits"], t["score"], -t["id"]))
    if frame_index % refresh_interval == 0:
        return full, "full_refresh", anchor["id"]
    crop_w, crop_h = max(1, width // zoom), max(1, height // zoom)
    x1, y1, x2, y2 = anchor["box"]
    left = min(max(round((x1 + x2) / 2 - crop_w / 2), 0), width - crop_w)
    top = min(max(round((y1 + y2) / 2 - crop_h / 2), 0), height - crop_h)
    return (left, top, left + crop_w, top + crop_h), "roi", anchor["id"]


def restore_boxes(boxes, region, width, height):
    """YOLO already reverses crop letterboxing; restore the crop translation."""
    restored = []
    for x1, y1, x2, y2, score in boxes:
        left, top = region[:2]
        restored.append([
            min(max(x1 + left, 0), width), min(max(y1 + top, 0), height),
            min(max(x2 + left, 0), width), min(max(y2 + top, 0), height), score,
        ])
    return restored
