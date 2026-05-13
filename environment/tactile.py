"""
Tactile sensing + gripper-safety wrapper.

Adapted from tactile-ril-env/ril_env/tactile.py for a single-process,
threaded teleop loop. Each A31301 ESP32 board streams 9 three-axis Hall
taxels over USB serial; one TactileSensor thread owns one serial port,
parses frames, and publishes the latest 9-taxel state. TactileSensors
bundles N of them and exposes the .safety() check used by the XArm
controller to clamp gripper closing when contact exceeds threshold.

Protocol reference: receive_a31301_stream_no_ros.py in ~/feng/tac_ws/...
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import serial  # pyserial
except ImportError:
    serial = None


logger = logging.getLogger(__name__)


# Lines that signal an ESP32 reboot/crash; parser goes silent until BEGIN_STREAM.
_REBOOT_PATTERNS = (
    "rst:0x", "boot:0x", "configsip:", "ets ",
    "Guru Meditation", "Backtrace:", "assert failed:", "abort()",
    "panic_handler", "LoadProhibited", "StoreProhibited",
    "IllegalInstruction", "IntegerDivideByZero",
)


@dataclass
class TactileConfig:
    """Configuration for one or more A31301 boards and the safety wrapper."""

    # One serial port per board. Default: two boards on /dev/ttyUSB{0,1}.
    ports: List[str] = field(default_factory=lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1"])
    baud: int = 115200
    n_taxels: int = 9

    # Reduce (n_sensors, n_taxels, 3) -> scalar:
    #   "max_abs_z"  max(|Bz|) across connected taxels
    #   "max_norm"   max(||xyz||) across connected taxels
    #   "sum_abs_z"  sum(|Bz|) across connected taxels
    safety_metric: str = "max_abs_z"
    # Trip threshold. Units match the device stream (raw counts by default).
    safety_threshold: float = 2000.0
    # Tactile data older than this is treated as unsafe (fail-safe).
    stale_after_sec: float = 0.2

    # Optional device-side configuration sent once at stream start.
    set_device_units: Optional[str] = None     # "raw" | "mt" | "g"
    set_device_rate_hz: Optional[int] = None   # clamped to [1, 100]

    # One-shot warning threshold for stale frames (ms). Independent of
    # safety_threshold above; this only controls a console nag.
    lag_warning_ms: int = 200

    verbose: bool = False

    # Optional per-cell baseline subtracted from xyz BEFORE the safety
    # metric is computed. Shape: (n_sensors, n_taxels, 3). When set, the
    # safety_threshold above is interpreted as a delta-from-idle threshold
    # (much smaller numerically than a raw threshold). When None, the
    # metric runs on raw values exactly as before (backwards compat).
    baseline: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def _is_reboot(line: str) -> bool:
    return any(p in line for p in _REBOOT_PATTERNS)


def _parse_sample(line: str):
    """Parse one ``S,ts,idx,addr,conn,x,y,z`` row. Returns tuple or None."""
    if not line.startswith("S,"):
        return None
    parts = line.split(",")
    if len(parts) != 8:
        return None
    try:
        return (
            int(parts[1]),    # ts_ms
            int(parts[2]),    # idx
            int(parts[4]),    # connected
            float(parts[5]),  # x
            float(parts[6]),  # y
            float(parts[7]),  # z
        )
    except Exception:
        return None


def compute_safety_metric(
    states: Sequence[Dict],
    metric: str,
    stale_after_sec: float,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[float, bool]:
    """Reduce per-sensor snapshots to a scalar.

    Returns (metric_value, all_fresh). ``all_fresh`` is False if any sensor
    has no connected taxels or has not produced a frame within
    ``stale_after_sec``. Callers should treat ``all_fresh=False`` as unsafe.

    If ``baseline`` is provided (shape (n_sensors, n_taxels, 3)), the
    per-cell idle field is subtracted from xyz before the metric reduces
    over cells. This makes the resulting metric a "change from idle"
    quantity, which is what carries the contact signal — raw |B_z| is
    dominated by each cell's static field and doesn't change much under
    contact for cells that idle near saturation.
    """
    now = time.time()
    per_sensor: List[float] = []
    all_fresh = True
    for i, st in enumerate(states):
        host_ts = float(st.get("host_timestamp", 0.0))
        conn = np.asarray(st.get("connected", np.zeros(0)), dtype=np.int32)
        xyz = np.asarray(st.get("xyz", np.zeros((0, 3))), dtype=np.float32)
        if (
            conn.size == 0
            or not np.any(conn)
            or host_ts <= 0.0
            or (now - host_ts) > stale_after_sec
        ):
            all_fresh = False
            continue
        mask = conn > 0
        # Subtract baseline if provided so the metric measures change-from-idle.
        if baseline is not None and i < baseline.shape[0]:
            xyz_use = xyz - np.asarray(baseline[i], dtype=np.float32)
        else:
            xyz_use = xyz
        if metric == "max_abs_z":
            per_sensor.append(float(np.max(np.abs(xyz_use[mask, 2]))))
        elif metric == "max_norm":
            per_sensor.append(float(np.max(np.linalg.norm(xyz_use[mask], axis=1))))
        elif metric == "sum_abs_z":
            per_sensor.append(float(np.sum(np.abs(xyz_use[mask, 2]))))
        else:
            raise ValueError(f"Unknown safety_metric: {metric!r}")
    if not per_sensor:
        return 0.0, False
    if metric == "sum_abs_z":
        return float(sum(per_sensor)), all_fresh
    return float(max(per_sensor)), all_fresh


def evaluate_safety(states: Sequence[Dict], config: TactileConfig) -> Tuple[float, bool]:
    """Returns (metric_value, is_safe_to_close).

    is_safe_to_close=False -> caller must NOT increase grasp closure. Stale
    or missing data forces unsafe (fail-safe). Opening is always allowed by
    the caller. If config.baseline is set, the metric is computed in
    "delta from idle" mode and config.safety_threshold is interpreted in
    the same units.
    """
    metric_val, fresh = compute_safety_metric(
        states, config.safety_metric, config.stale_after_sec, baseline=config.baseline,
    )
    if not fresh:
        return metric_val, False
    return metric_val, metric_val <= config.safety_threshold


# ---------------------------------------------------------------------------
# Per-board reader thread
# ---------------------------------------------------------------------------

class TactileSensor(threading.Thread):
    """One thread per A31301 board.

    Owns the serial port, parses frames, and publishes the latest 9-taxel
    state under a lock. ``open_failed`` is set if the port couldn't be
    opened — callers can use it to disable tactile cleanly.
    """

    def __init__(self, port: str, config: TactileConfig, name: Optional[str] = None):
        super().__init__(daemon=True, name=name or f"TactileSensor-{port}")
        if serial is None:
            raise ImportError(
                "pyserial is required for TactileSensor; "
                "install with `pip install pyserial`."
            )
        self.port = port
        self.config = config
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.open_failed = False

        n = config.n_taxels
        self._lock = threading.Lock()
        self._latest = {
            "xyz": np.zeros((n, 3), dtype=np.float32),
            "connected": np.zeros(n, dtype=np.int32),
            "device_ts_ms": np.int64(0),
            "host_timestamp": 0.0,  # 0.0 -> always stale until first publish
        }

        if config.verbose:
            logger.setLevel(logging.DEBUG)

    def get_state(self) -> Dict:
        """Return a copy of the latest 9-taxel frame as a dict."""
        with self._lock:
            return {
                "xyz": self._latest["xyz"].copy(),
                "connected": self._latest["connected"].copy(),
                "device_ts_ms": int(self._latest["device_ts_ms"]),
                "host_timestamp": float(self._latest["host_timestamp"]),
            }

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        # Give the OS a moment to either open the port or fail.
        self.ready_event.wait(timeout=0.5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Worker entry point
    # ------------------------------------------------------------------

    def run(self):
        cfg = self.config
        try:
            ser = serial.Serial(
                self.port, cfg.baud, timeout=1, rtscts=False, dsrdtr=False
            )
            # Hold DTR/RTS so opening doesn't reset the ESP32.
            try:
                ser.setDTR(True)
                ser.setRTS(True)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[tactile {self.name}] open {self.port} failed: {e}")
            self.open_failed = True
            self.ready_event.set()
            return

        def send_cmd(line: str):
            try:
                ser.write((line + "\n").encode())
                ser.flush()
            except Exception as e:
                logger.error(f"[tactile {self.name}] cmd '{line}' failed: {e}")

        if cfg.set_device_units:
            send_cmd(f"CMD,SET,UNITS,{cfg.set_device_units}")
        if cfg.set_device_rate_hz is not None:
            hz = max(1, min(100, int(cfg.set_device_rate_hz)))
            send_cmd(f"CMD,SET,RATE_HZ,{hz}")
        send_cmd("CMD,GET,STATE")

        self.ready_event.set()

        n = cfg.n_taxels
        frame_xyz = np.zeros((n, 3), dtype=np.float32)
        frame_conn = np.zeros(n, dtype=np.int32)
        seen = np.zeros(n, dtype=bool)
        current_ts: Optional[int] = None
        reboot_detected = False

        try:
            while not self.stop_event.is_set():
                try:
                    raw = ser.readline().decode(errors="ignore")
                except Exception as e:
                    logger.error(f"[tactile {self.name}] read error: {e}")
                    break
                if not raw:
                    continue
                line = raw.strip()
                if not line:
                    continue

                if _is_reboot(line):
                    if not reboot_detected:
                        logger.warning(
                            f"[tactile {self.name}] device reboot/crash: {line}"
                        )
                        reboot_detected = True
                    current_ts = None
                    seen[:] = False
                    continue
                if reboot_detected:
                    if line.startswith("BEGIN_STREAM"):
                        logger.info(f"[tactile {self.name}] recovered: {line}")
                        reboot_detected = False
                        current_ts = None
                        seen[:] = False
                    continue

                if line.startswith("BEGIN_STREAM") or line.startswith("END_STREAM"):
                    continue

                parsed = _parse_sample(line)
                if parsed is None:
                    continue
                ts_ms, idx, conn, x, y, z = parsed
                if not (0 <= idx < n):
                    continue

                if current_ts is None:
                    current_ts = ts_ms

                # New ts_ms -> finalize the (possibly partial) previous frame.
                if ts_ms != current_ts:
                    self._publish(frame_xyz, frame_conn, current_ts)
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    current_ts = ts_ms

                frame_xyz[idx] = (x, y, z)
                frame_conn[idx] = conn
                seen[idx] = True

                if seen.all():
                    self._publish(frame_xyz, frame_conn, current_ts)
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    current_ts = None
        finally:
            try:
                ser.setDTR(True)
                ser.setRTS(True)
                ser.close()
            except Exception:
                pass

    def _publish(self, xyz: np.ndarray, conn: np.ndarray, ts_ms: int):
        with self._lock:
            self._latest["xyz"] = xyz.copy()
            self._latest["connected"] = conn.copy()
            self._latest["device_ts_ms"] = np.int64(ts_ms)
            self._latest["host_timestamp"] = time.time()


# ---------------------------------------------------------------------------
# Multi-board bundle
# ---------------------------------------------------------------------------

class TactileSensors:
    """Bundle of N TactileSensor threads, one per port in config.ports.

    Use as a context manager. Pass the instance to XArm(..., tactile=...)
    to wire up gripper safety; the controller's public API is unchanged.
    """

    def __init__(self, config: TactileConfig, names: Optional[List[str]] = None):
        self.config = config
        labels = names or [f"sensor{i}" for i in range(len(config.ports))]
        self.sensors: List[TactileSensor] = [
            TactileSensor(port, config, name=lbl)
            for port, lbl in zip(config.ports, labels)
        ]

    # --- context manager -------------------------------------------------
    def __enter__(self):
        for s in self.sensors:
            s.start()
        # Give every sensor a chance to open (or fail) before we return.
        for s in self.sensors:
            s.ready_event.wait(timeout=0.5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self.sensors:
            s.stop()
        for s in self.sensors:
            s.join(timeout=2.0)

    # --- introspection ---------------------------------------------------
    @property
    def all_open(self) -> bool:
        return all(not s.open_failed for s in self.sensors)

    @property
    def any_open(self) -> bool:
        return any(not s.open_failed for s in self.sensors)

    def get_latest(self) -> List[Dict]:
        """Latest per-sensor state, one dict per board (same order as config.ports)."""
        return [s.get_state() for s in self.sensors]

    def safety(self) -> Tuple[float, bool]:
        """(metric_value, is_safe_to_close) using the configured metric/threshold."""
        return evaluate_safety(self.get_latest(), self.config)
