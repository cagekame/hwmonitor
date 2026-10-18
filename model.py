"""
Data model shared by the reader and the UI.
- Snapshot: one typed reading of all sensors (produced by hwreader).
- SensorState: latest snapshot + rolling history + derived stats (max/avg, fan peaks).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

COMPONENTS = ("cpu", "gpu", "mobo")


def num(v) -> Optional[float]:
    """float(v), or None for None / NaN / garbage."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


@dataclass
class Snapshot:
    timestamp: float = field(default_factory=time.time)
    # temperatures (°C)
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    mobo_temp: Optional[float] = None
    # load (%)
    cpu_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    ram_usage: Optional[float] = None
    # memory (MB)
    ram_used_mb: Optional[float] = None
    ram_free_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    # fans (RPM)
    fan_cpu_rpm: Optional[float] = None
    fan_gpu_rpm: Optional[float] = None
    fan_pump_rpm: Optional[float] = None
    fans_sys: Dict[str, Optional[float]] = field(default_factory=dict)
    # display names of the components ("cpu" / "gpu" / "mobo")
    labels: Dict[str, str] = field(default_factory=dict)

    def temp(self, component: str) -> Optional[float]:
        return num(getattr(self, f"{component}_temp"))


@dataclass
class FanReading:
    name: str
    key: str            # stable id used for peak tracking
    rpm: Optional[float]
    is_pump: bool = False


class SensorState:
    """Latest snapshot + history. Pure data, no UI."""

    HISTORY_FIELDS = ("cpu_temp", "gpu_temp", "mobo_temp", "cpu_usage", "gpu_usage", "ram_usage")

    def __init__(self, history_len: int = 300):
        self.latest = Snapshot()
        self.history: Dict[str, Deque[Optional[float]]] = {k: deque(maxlen=history_len) for k in self.HISTORY_FIELDS}
        self._fan_peak: Dict[str, float] = {}

    def resize(self, history_len: int):
        for k, d in self.history.items():
            if d.maxlen != history_len:
                self.history[k] = deque(d, maxlen=history_len)

    def update(self, snap: Snapshot):
        self.latest = snap
        for k in self.HISTORY_FIELDS:
            self.history[k].append(num(getattr(snap, k)))
        for f in self.fans():
            if f.rpm is not None:
                self._fan_peak[f.key] = max(self._fan_peak.get(f.key, 0.0), f.rpm)

    # ---------- derived ----------
    def temp_stats(self, component: str) -> Tuple[Optional[float], Optional[float]]:
        """(max, avg) temperature over the history window."""
        vals = [v for v in self.history[f"{component}_temp"] if v is not None]
        if not vals:
            return None, None
        return max(vals), sum(vals) / len(vals)

    def fans(self) -> List[FanReading]:
        """CPU / GPU / Pump first (only if present), then system fans."""
        s = self.latest
        fixed = [FanReading("CPU", "cpu", num(s.fan_cpu_rpm)),
                 FanReading("GPU", "gpu", num(s.fan_gpu_rpm)),
                 FanReading("Pump", "pump", num(s.fan_pump_rpm), is_pump=True)]
        sys_fans = [FanReading(str(n), f"sys:{n}", num(v)) for n, v in (s.fans_sys or {}).items()]
        out = [f for f in fixed if f.rpm is not None] + sys_fans
        return out or fixed

    def fan_fraction(self, fan: FanReading, full_scale: float) -> Optional[float]:
        """rpm / max(full_scale, observed peak) — bars never overflow."""
        if fan.rpm is None:
            return None
        top = max(full_scale, self._fan_peak.get(fan.key, 0.0))
        return fan.rpm / top if top > 0 else None
