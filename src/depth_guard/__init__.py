"""Depth-Guard 3D spatial occupancy monitor."""

from .engine import analyze_frame
from .io import load_rules, load_vision_feed
from .models import ObjectStatus

__all__ = ["ObjectStatus", "analyze_frame", "load_rules", "load_vision_feed"]

__version__ = "1.0.0"
