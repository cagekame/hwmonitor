"""
UI dashboard (Tkinter) that reads layout/theme from hwconfig.json and live data from hwreader.HwinfoReader.
- Pure presentation for sensors (no HWiNFO parsing here).
- Customizable via `hwconfig.json` placed next to this file.
- Fullscreen monitor selection by position: left/right/top/bottom/primary/index.
- Shows fan info next to usage bars:
  * CPU bar → "CPUFan: <RPM>"
  * RAM bar → "Pump: <RPM>"  (reads fan_pump_rpm from HwinfoReader; falls back to fans.sys if missing)
  * GPU bar → "GpuFan: <RPM>"
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
import queue
import time
from typing import Deque, Dict, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== CONFIG =====================
DEFAULT_CONFIG = {
    "window": {
        "mode": "fullscreen",  # "fullscreen" | "windowed"
        "geometry": "1280x720+100+100",
        "monitor": {
            "select": "right",  # left|right|top|bottom|primary|index
            "index": 1,
            "index_base": 1,
            "safe_margins": {"l": 0, "t": 0, "r": 0, "b": 0},
        },
    },
    "theme": {
        "bg": "#0b0e13",
        "fg": "#e5eef9",
        "sub": "#9fb0c6",
        "grid": "#16202f",
        "track": "#151922",
        "font_family": "Consolas",
        "ui_scale": 1.0,
    },
    "fonts": {
        "clock_pt": 40,
        "bar_label_pt": 18,
        "ring_value_scale": 0.28,
        "ring_label_scale": 0.06,
    },
    "poll": {"interval_sec": 1.0, "history_seconds": 300},
    "layout": {"outer_padding_px": 20, "inner_gap_px": 10, "graph_height_px": 120},
    "rings": {
        "cpu": {"vmax": 100, "label_offset": 1.10},
        "mobo": {"vmax": 95, "label_offset": 1.10},
        "gpu": {"vmax": 110, "label_offset": 1.10},
    },
    "graphs": {"cpu": {"max_value": 110}, "mobo": {"max_value": 100}, "gpu": {"max_value": 115}},
    "bars": {"height": 54, "labels": {"cpu": "CPU Usage", "ram": "RAM Usage", "gpu": "GPU Usage"}},
    "labels": {"override": {"cpu": None, "mobo": None, "gpu": None}},
    "hotkeys": {"fullscreen_toggle": "F11", "rescan_hwinfo": "R"},
}

def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_config(path: str = "hwconfig.json") -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    # se il path non è assoluto, referenzialo alla cartella dello script
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = json.load(f)
            _deep_merge(cfg, user)
        except Exception:
            pass
    return cfg

# ===================== THEME (defaults, will be overridden by cfg) =====================
BG = DEFAULT_CONFIG["theme"]["bg"]
FG = DEFAULT_CONFIG["theme"]["fg"]
SUB = DEFAULT_CONFIG["theme"]["sub"]
GRID = DEFAULT_CONFIG["theme"]["grid"]
TRACK = DEFAULT_CONFIG["theme"]["track"]
FONT_FAMILY = DEFAULT_CONFIG["theme"]["font_family"]

# ===================== MONITOR PICK (Windows only; safe no-op on others) =====================
try:
    import ctypes as ct
    from ctypes import wintypes

    class RECT(ct.Structure):
        _fields_ = [("left", ct.c_long), ("top", ct.c_long), ("right", ct.c_long), ("bottom", ct.c_long)]

    class MONITORINFOEXW(ct.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    PMONITORENUMPROC = ct.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ct.POINTER(RECT), wintypes.LPARAM)
    _u32 = ct.windll.user32
    _u32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ct.POINTER(RECT), PMONITORENUMPROC, wintypes.LPARAM]
    _u32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ct.POINTER(MONITORINFOEXW)]

    def list_monitors() -> list[dict]:
        mons = []
        def _cb(hMon, hdc, lprc, lparam):
            info = MONITORINFOEXW()
            info.cbSize = ct.sizeof(MONITORINFOEXW)
            if _u32.GetMonitorInfoW(hMon, ct.byref(info)):
                x, y = info.rcMonitor.left, info.rcMonitor.top
                w = info.rcMonitor.right - info.rcMonitor.left
                h = info.rcMonitor.bottom - info.rcMonitor.top
                primary = bool(info.dwFlags & 1)
                mons.append({"device": info.szDevice, "x": x, "y": y, "w": w, "h": h, "primary": primary})
            return True
        _u32.EnumDisplayMonitors(0, None, PMONITORENUMPROC(_cb), 0)
        return mons
except Exception:
    def list_monitors() -> list[dict]:
        root = tk.Tk()
        w = root.winfo_screenwidth(); h = root.winfo_screenheight()
        root.destroy()
        return [{"device": "primary", "x": 0, "y": 0, "w": w, "h": h, "primary": True}]

def _pick_monitor(cfg_monitor: dict) -> dict:
    mons = list_monitors()
    if not mons:
        return {"x": 0, "y": 0, "w": 800, "h": 600}
    sel = (cfg_monitor.get("select") or "primary").lower()
    if sel == "primary":
        for m in mons:
            if m.get("primary"):
                return m
        return mons[0]
    if sel == "left":
        return min(mons, key=lambda m: (m["x"], m["y"]))
    if sel == "right":
        return max(mons, key=lambda m: (m["x"], m["y"]))
    if sel == "top":
        return min(mons, key=lambda m: (m["y"], m["x"]))
    if sel == "bottom":
        return max(mons, key=lambda m: (m["y"], m["x"]))
    if sel == "index":
        base = int(cfg_monitor.get("index_base", 1))
        idx = int(cfg_monitor.get("index", 1)) - base
        if 0 <= idx < len(mons):
            return mons[idx]
        return mons[-1]
    return mons[0]

# ===================== WIDGETS =====================
class RingGauge(tk.Canvas):
    def __init__(self, parent, vmax=100.0, value_scale=0.28, label_scale=0.07, label_offset=1.08, font_family: str = "Consolas"):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.maxv = float(vmax)
        self.value_scale = value_scale
        self.label_scale = label_scale
        self.label_offset = label_offset
        self.font_family = font_family
        self.arc_id = self.val_id = self.bottom_label = None
        self.bind("<Configure>", self._resize)

    def _resize(self, _e=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        size = min(w, h) * 0.8
        cx, cy = w // 2, h // 2
        r = size / 2
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=TRACK, width=max(6, int(size * 0.04)))
        self.arc_id = self.create_arc(
            cx - r, cy - r, cx + r, cy + r, start=90, extent=0,
            style="arc", outline="#ffffff", width=max(10, int(size * 0.06))
        )
        self.val_id = self.create_text(
            cx, cy, text="--°", fill=FG, font=(self.font_family, int(size * self.value_scale), "bold")
        )
        self.bottom_label = self.create_text(
            cx, cy + r * self.label_offset, text="", fill=SUB,
            font=(self.font_family, int(size * self.label_scale)), anchor="n"
        )

    def set(self, value: Optional[float], label_text: str):
        try:
            v = float(value) if value is not None else None
        except Exception:
            v = None
        if v is None or math.isnan(v):
            self.itemconfigure(self.arc_id, extent=0)
            self.itemconfigure(self.val_id, text="--°")
        else:
            frac = max(0.0, min(1.0, v / max(1e-9, self.maxv)))
            self.itemconfigure(self.arc_id, extent=360 * frac)
            self.itemconfigure(self.val_id, text=f"{v:.0f}°")
        self.itemconfigure(self.bottom_label, text=label_text or "")

class TrendGraph(tk.Canvas):
    PAD_L = 24; PAD_R = 16; PAD_T = 18; PAD_B = 18
    def __init__(self, parent, max_seconds=300, max_value=120.0, height_px=120):
        super().__init__(parent, bg=BG, highlightthickness=0, height=height_px)
        self.max_seconds = max_seconds
        self.max_value = float(max_value)
        self.points: Deque[Tuple[float, Optional[float]]] = deque()
        self.bind("<Configure>", self._redraw)

    def push(self, value: Optional[float]):
        t = time.time()
        try:
            v = float(value) if value is not None else None
        except Exception:
            v = None
        self.points.append((t, v))
        cutoff = t - self.max_seconds
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()
        self._redraw()

    def _redraw(self, _e=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 4 or h <= 4:
            return
        x0, y0, x1, y1 = self.PAD_L, self.PAD_T, w - self.PAD_R, h - self.PAD_B
        iw, ih = max(1, x1 - x0), max(1, y1 - y0)
        self.create_rectangle(x0, y0, x1, y1, outline=GRID, width=1)
        for xfrac in (0.25, 0.5, 0.75):
            x = int(x0 + iw * xfrac)
            self.create_line(x, y0, x, y1, fill=GRID)
        self.create_line(x0, int(y0 + ih * 0.5), x1, int(y0 + ih * 0.5), fill=GRID)
        if len(self.points) < 2:
            return
        t_now = self.points[-1][0]
        t_min = t_now - self.max_seconds
        path = []
        for (t, v) in self.points:
            if t < t_min:
                continue
            x = int(x0 + (t - t_min) / self.max_seconds * (iw - 1))
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if len(path) > 1:
                    self.create_line(path, fill="#ffffff", width=2, smooth=True)
                path = []
                continue
            y = int(y1 - (max(0.0, min(self.max_value, v)) / self.max_value) * (ih - 1))
            path.append((x, y))
        if len(path) > 1:
            self.create_line(path, fill="#ffffff", width=2, smooth=True)

class LoadBar(tk.Canvas):
    PAD_H = 24; PAD_V = 10
    def __init__(self, parent, height=50, label_text="Usage", label_pt=18, font_family: str = "Consolas"):
        super().__init__(parent, bg=BG, highlightthickness=0, height=height)
        self.value: Optional[float] = None
        self.label_text = label_text
        self.label_pt = int(label_pt)
        self.font_family = font_family
        # extra label shown at the RIGHT side (e.g., CPUFan: 1200 RPM)
        self._extra_label: Optional[str] = None
        self._extra_value: Optional[float] = None
        self._extra_unit: str = "RPM"
        self.bind("<Configure>", self._redraw)

    def set(self, value: Optional[float]):
        self.value = value
        self._redraw()

    def set_extra(self, label: Optional[str], value: Optional[float], unit: str = "RPM"):
        self._extra_label = (label or None)
        self._extra_value = (None if value is None else float(value))
        self._extra_unit = unit
        self._redraw()

    def _format_extra(self) -> Optional[str]:
        if not self._extra_label:
            return None
        if self._extra_value is None or (isinstance(self._extra_value, float) and math.isnan(self._extra_value)):
            return f"{self._extra_label}: -- {self._extra_unit}"
        try:
            v = int(round(self._extra_value))
        except Exception:
            return f"{self._extra_label}: -- {self._extra_unit}"
        return f"{self._extra_label}: {v} {self._extra_unit}"

    def _redraw(self, _e=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        txt = f"{self.label_text}: --%"
        frac = 0.0
        if self.value is not None:
            try:
                v = float(self.value)
                v = max(0.0, min(100.0, v))
                frac = v / 100.0
                txt = f"{self.label_text}: {v:.0f}%"
            except Exception:
                pass
        # Left main label
        self.create_text(self.PAD_H, self.PAD_V, anchor="w", text=txt, fill=FG, font=(self.font_family, self.label_pt, "bold"))
        # Right extra label
        extra = self._format_extra()
        if extra:
            self.create_text(w - self.PAD_H, self.PAD_V, anchor="e", text=extra, fill=SUB, font=(self.font_family, int(self.label_pt*0.9)))
        # Track
        x0, x1 = self.PAD_H, w - self.PAD_H
        y0, y1 = h - (self.PAD_V + 30), h - self.PAD_V
        self.create_rectangle(x0, y0, x1, y1, outline=TRACK, width=2)
        xf = x0 + int((x1 - x0) * frac)
        self.create_rectangle(x0 + 2, y0 + 2, max(x0 + 2, xf - 2), y1 - 2, outline="", fill="#ffffff")

# ===================== UI APP =====================
@dataclass
class UiConfig:
    max_cpu_temp: float = 100.0
    max_mobo_temp: float = 95.0
    max_gpu_temp: float = 110.0
    history_seconds: int = 300
    graph_height: int = 120
    ui_scale: float = 1.0
    clock_pt: int = 40
    bar_label_pt: int = 18
    ring_value_scale: float = 0.28
    ring_label_scale: float = 0.06
    ring_label_offset_cpu: float = 1.10
    ring_label_offset_mobo: float = 1.10
    ring_label_offset_gpu: float = 1.10
    font_family: str = "Consolas"
    outer_padding_px: int = 20
    inner_gap_px: int = 10
    bars_height: int = 54

class DashboardTk(tk.Tk):
    """Presentation-only Tkinter app. Use `post_snapshot(snapshot)` to feed data."""

    def __init__(self, cfg: dict):
        super().__init__()
        # Theme
        global BG, FG, SUB, GRID, TRACK, FONT_FAMILY
        th = cfg.get("theme", {})
        BG = th.get("bg", BG); FG = th.get("fg", FG); SUB = th.get("sub", SUB)
        GRID = th.get("grid", GRID); TRACK = th.get("track", TRACK)
        FONT_FAMILY = th.get("font_family", FONT_FAMILY)
        ui_scale = float(th.get("ui_scale", 1.0))

        # Derived UiConfig
        fonts = cfg.get("fonts", {})
        rings = cfg.get("rings", {})
        graphs = cfg.get("graphs", {})
        layout = cfg.get("layout", {})
        bars = cfg.get("bars", {})
        poll = cfg.get("poll", {})

        self.ui_cfg = UiConfig(
            max_cpu_temp=float(rings.get("cpu", {}).get("vmax", 100)),
            max_mobo_temp=float(rings.get("mobo", {}).get("vmax", 95)),
            max_gpu_temp=float(rings.get("gpu", {}).get("vmax", 110)),
            history_seconds=int(poll.get("history_seconds", 300)),
            graph_height=int(layout.get("graph_height_px", 120) * ui_scale),
            ui_scale=ui_scale,
            clock_pt=int(fonts.get("clock_pt", 40) * ui_scale),
            bar_label_pt=int(fonts.get("bar_label_pt", 18) * ui_scale),
            ring_value_scale=float(fonts.get("ring_value_scale", 0.28)),
            ring_label_scale=float(fonts.get("ring_label_scale", 0.06)),
            ring_label_offset_cpu=float(rings.get("cpu", {}).get("label_offset", 1.10)),
            ring_label_offset_mobo=float(rings.get("mobo", {}).get("label_offset", 1.10)),
            ring_label_offset_gpu=float(rings.get("gpu", {}).get("label_offset", 1.10)),
            font_family=FONT_FAMILY,
            outer_padding_px=int(layout.get("outer_padding_px", 20) * ui_scale),
            inner_gap_px=int(layout.get("inner_gap_px", 10) * ui_scale),
            bars_height=int(bars.get("height", 54) * ui_scale),
        )
        self.graph_cfg = {
            "cpu": float(graphs.get("cpu", {}).get("max_value", self.ui_cfg.max_cpu_temp + 10)),
            "mobo": float(graphs.get("mobo", {}).get("max_value", self.ui_cfg.max_mobo_temp + 5)),
            "gpu": float(graphs.get("gpu", {}).get("max_value", self.ui_cfg.max_gpu_temp + 5)),
        }
        self.bar_labels = bars.get("labels", {"cpu": "CPU Usage", "ram": "RAM Usage", "gpu": "GPU Usage"})

        # Window base
        self.configure(bg=BG)
        self.title("Dashboard — CPU / PCH / GPU")

        # State
        self.labels: Dict[str, str] = {"cpu": "CPU", "mobo": "Motherboard / PCH", "gpu": "GPU"}
        overrides = (cfg.get("labels", {}).get("override", {}) or {})
        for k in ("cpu", "mobo", "gpu"):
            if overrides.get(k):
                self.labels[k] = str(overrides[k])
        self._q: "queue.Queue[dict]" = queue.Queue()

        # Header clock
        header = tk.Frame(self, bg=BG, height=int(self.ui_cfg.clock_pt * 2))
        header.pack(side="top", fill="x")
        self.clock_lbl = tk.Label(header, text="", bg=BG, fg=FG, font=(FONT_FAMILY, self.ui_cfg.clock_pt, "bold"))
        self.clock_lbl.pack(anchor="center", pady=int(self.ui_cfg.inner_gap_px))

        # Grid container 3x3
        container = tk.Frame(self, bg=BG)
        container.pack(expand=True, fill="both", padx=self.ui_cfg.outer_padding_px, pady=self.ui_cfg.outer_padding_px)
        for c in (0, 1, 2):
            container.grid_columnconfigure(c, weight=1)
        container.grid_rowconfigure(0, weight=3)
        container.grid_rowconfigure(1, weight=2)
        container.grid_rowconfigure(2, weight=0)

        pad = self.ui_cfg.inner_gap_px

        # Ring gauges
        self.gauge_cpu = RingGauge(container, vmax=self.ui_cfg.max_cpu_temp,
                                   value_scale=self.ui_cfg.ring_value_scale,
                                   label_scale=self.ui_cfg.ring_label_scale,
                                   label_offset=self.ui_cfg.ring_label_offset_cpu,
                                   font_family=self.ui_cfg.font_family)
        self.gauge_mobo = RingGauge(container, vmax=self.ui_cfg.max_mobo_temp,
                                    value_scale=self.ui_cfg.ring_value_scale,
                                    label_scale=self.ui_cfg.ring_label_scale,
                                    label_offset=self.ui_cfg.ring_label_offset_mobo,
                                    font_family=self.ui_cfg.font_family)
        self.gauge_gpu = RingGauge(container, vmax=self.ui_cfg.max_gpu_temp,
                                   value_scale=self.ui_cfg.ring_value_scale,
                                   label_scale=self.ui_cfg.ring_label_scale,
                                   label_offset=self.ui_cfg.ring_label_offset_gpu,
                                   font_family=self.ui_cfg.font_family)
        self.gauge_cpu.grid(row=0, column=0, sticky="nsew", padx=pad, pady=pad)
        self.gauge_mobo.grid(row=0, column=1, sticky="nsew", padx=pad, pady=pad)
        self.gauge_gpu.grid(row=0, column=2, sticky="nsew", padx=pad, pady=pad)

        # Sparklines
        self.graph_cpu = TrendGraph(container, max_seconds=self.ui_cfg.history_seconds,
                                    max_value=self.graph_cfg["cpu"], height_px=self.ui_cfg.graph_height)
        self.graph_mobo = TrendGraph(container, max_seconds=self.ui_cfg.history_seconds,
                                     max_value=self.graph_cfg["mobo"], height_px=self.ui_cfg.graph_height)
        self.graph_gpu = TrendGraph(container, max_seconds=self.ui_cfg.history_seconds,
                                    max_value=self.graph_cfg["gpu"], height_px=self.ui_cfg.graph_height)
        self.graph_cpu.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(2, pad))
        self.graph_mobo.grid(row=1, column=1, sticky="nsew", padx=pad, pady=(2, pad))
        self.graph_gpu.grid(row=1, column=2, sticky="nsew", padx=pad, pady=(2, pad))

        # Usage bars
        self.cpu_usage_bar = LoadBar(container, height=self.ui_cfg.bars_height, label_text=self.bar_labels.get("cpu", "CPU Usage"), label_pt=self.ui_cfg.bar_label_pt, font_family=self.ui_cfg.font_family)
        self.ram_usage_bar = LoadBar(container, height=self.ui_cfg.bars_height, label_text=self.bar_labels.get("ram", "RAM Usage"), label_pt=self.ui_cfg.bar_label_pt, font_family=self.ui_cfg.font_family)
        self.gpu_usage_bar = LoadBar(container, height=self.ui_cfg.bars_height, label_text=self.bar_labels.get("gpu", "GPU Usage"), label_pt=self.ui_cfg.bar_label_pt, font_family=self.ui_cfg.font_family)
        self.cpu_usage_bar.grid(row=2, column=0, sticky="ew", padx=pad, pady=(0, pad))
        self.ram_usage_bar.grid(row=2, column=1, sticky="ew", padx=pad, pady=(0, pad))
        self.gpu_usage_bar.grid(row=2, column=2, sticky="ew", padx=pad, pady=(0, pad))

        # Hotkeys
        hk = cfg.get("hotkeys", {})
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind(f"<{hk.get('fullscreen_toggle', 'F11')}>", lambda e: self._toggle_fullscreen())

        # Start UI tickers
        self.after(200, self._tick_clock)
        self.after(100, self._drain_queue)

    # ---------- Public API ----------
    def set_labels(self, labels: Dict[str, str]):
        self.labels.update({k: v for k, v in labels.items() if k in ("cpu", "mobo", "gpu") and v})
        # Reflect immediately
        self.gauge_cpu.set(None, self.labels["cpu"])
        self.gauge_mobo.set(None, self.labels["mobo"])
        self.gauge_gpu.set(None, self.labels["gpu"])

    def post_snapshot(self, snapshot: Dict):
        self._q.put(snapshot)

    # ---------- Internals ----------
    def _tick_clock(self):
        self.clock_lbl.configure(text=f"{datetime.now():%H:%M:%S   %d.%m.%Y}", fg=FG)
        self.after(200, self._tick_clock)

    def _drain_queue(self):
        drained = 0
        while drained < 60:
            try:
                snap = self._q.get_nowait()
            except queue.Empty:
                break
            self._apply_snapshot(snap)
            drained += 1
        self.after(100, self._drain_queue)

    def _apply_snapshot(self, snap: Dict):
        if isinstance(snap.get('labels'), dict):
            self.set_labels(snap['labels'])

        cpu_t = snap.get('cpu_temp'); pch_t = snap.get('pch_temp'); gpu_t = snap.get('gpu_temp')
        self.gauge_cpu.set(cpu_t, self.labels['cpu'])
        self.gauge_mobo.set(pch_t, self.labels['mobo'])
        self.gauge_gpu.set(gpu_t, self.labels['gpu'])
        self.graph_cpu.push(cpu_t); self.graph_mobo.push(pch_t); self.graph_gpu.push(gpu_t)

        # main values
        self.cpu_usage_bar.set(snap.get('cpu_usage'))
        self.ram_usage_bar.set(snap.get('ram_usage'))
        self.gpu_usage_bar.set(snap.get('gpu_usage'))

        # extra: fans
        cpu_fan = snap.get('fan_cpu_rpm')
        gpu_fan = snap.get('fan_gpu_rpm')
        pump_rpm = snap.get('fan_pump_rpm')  # <-- usa il campo diretto dal reader

        # Fallback: se non presente, cerca "pump/pompa" dentro fans.sys
        if pump_rpm is None:
            try:
                fans_sys = (snap.get('fans') or {}).get('sys') or {}
                for name, val in fans_sys.items():
                    n = (name or '').lower()
                    if any(tok in n for tok in ('pump', 'pompa')):
                        pump_rpm = val
                        break
            except Exception:
                pass

        self.cpu_usage_bar.set_extra('CPU Fan', cpu_fan, 'RPM')
        self.ram_usage_bar.set_extra('Pump', pump_rpm, 'RPM')
        self.gpu_usage_bar.set_extra('Gpu Fan', gpu_fan, 'RPM')

    def _toggle_fullscreen(self):
        cur = self.overrideredirect()
        self.overrideredirect(not cur)
        self.update_idletasks()

# ===================== LAUNCHER (reads config + hwreader) =====================
if __name__ == "__main__":
    cfg = load_config("hwconfig.json")

    # Prepare window and monitor placement
    win_cfg = cfg.get("window", {})
    monitor_cfg = (win_cfg.get("monitor") or {})
    mon = _pick_monitor(monitor_cfg)

    app = DashboardTk(cfg)

    # Place window
    if (win_cfg.get("mode") or "fullscreen").lower() == "fullscreen":
        # Apply margins
        margins = monitor_cfg.get("safe_margins", {}) or {}
        l = int(margins.get("l", 0)); t = int(margins.get("t", 0)); r = int(margins.get("r", 0)); b = int(margins.get("b", 0))
        x = mon["x"] + l; y = mon["y"] + t
        w = max(100, mon["w"] - (l + r)); h = max(100, mon["h"] - (t + b))
        app.geometry(f"{w}x{h}+{x}+{y}")
        try:
            app._toggle_fullscreen()
        except Exception:
            try:
                app.attributes('-fullscreen', True)
            except Exception:
                pass
    else:
        app.geometry(win_cfg.get("geometry", "1280x720+100+100"))

    # Data source (hwreader)
    poll_ms = int(float(cfg.get("poll", {}).get("interval_sec", 1.0)) * 1000)
    try:
        from hwreader import HwinfoReader
        rdr = HwinfoReader()
        rdr.open(); rdr.scan()
        try:
            app.set_labels(rdr.labels())
        except Exception:
            pass

        def tick():
            try:
                snap = rdr.read_snapshot()
                app.post_snapshot(snap)
            except Exception as e:
                app.clock_lbl.configure(text=str(e)[:120], fg="#ff7a1a")
            finally:
                app.after(poll_ms, tick)

        def on_close():
            try:
                rdr.close()
            finally:
                app.destroy()

        hk = cfg.get("hotkeys", {})
        rescan_key = hk.get("rescan_hwinfo", "R")
        app.bind(f"<{rescan_key}>", lambda e: rdr.scan())
        app.protocol("WM_DELETE_WINDOW", on_close)
        tick()
    except Exception as e:
        app.clock_lbl.configure(text=f"hwreader error: {e}", fg="#ff7a1a")

    app.mainloop()
