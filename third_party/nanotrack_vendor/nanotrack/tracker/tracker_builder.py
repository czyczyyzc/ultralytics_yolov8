from nanotrack.core.config import cfg
from nanotrack.tracker.nano_tracker import NanoTracker

TRACKS = {"NanoTracker": NanoTracker}


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
