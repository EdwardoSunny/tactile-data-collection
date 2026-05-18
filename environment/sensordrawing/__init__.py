"""
Vendored copy of /Users/yluo/Documents/Research/sensordrawing.

Provides kinematics-aware tactile sensor visualization. See draw_sensors.py
for the SensorDrawer entry point and normalizer.py for SensorNormalizer.
"""
from .draw_sensors import SensorDrawer
from .normalizer import SensorNormalizer

__all__ = ["SensorDrawer", "SensorNormalizer"]
