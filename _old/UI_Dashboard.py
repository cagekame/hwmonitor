"""
UI dashboard (Tkinter) that reads layout/theme from hwconfig.json and live data from hwreader.HwinfoReader.
- Pure presentation for sensors (no HWiNFO parsing here).
- Customizable via `hwconfig.json` placed next to this file.
- Fullscreen monitor selection by position: left/right/top/bottom/primary/index.
- "Panel" layout drawn on a single Canvas:
  * top row:    CPU / GPU panels (segmented temperature gauge + Load / Fan / Max / Avg rows)
  * bottom row: RAM / FANS / MOBO / SYS panels
"""

from __future__ import annotations

import json
import os
import platform
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from datetime import datetime
import math
import queue
import time
from typing import Deque, Dict, List, Optional, Tuple

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
        "hot": "#ff5a4f",        # color used when a temperature is near its max
        "hot_ratio": 0.85,       # fraction of vmax above which "hot" kicks in
        "font_family": "Consolas",
        "display_font": "Bahnschrift",  # panel font (falls back to font_family if missing)
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
    "fans": {"max_rpm": 2000, "pump_max_rpm": 3500},
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
HOT = DEFAULT_CONFIG["theme"]["hot"]
DIM = GRID  # recomputed from fg/bg at startup
FONT_FAMILY = DEFAULT_CONFIG["theme"]["font_family"]
OFFLINE = "#ff7a1a"

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

# ===================== SYSTEM INFO HELPERS =====================
def _display_info(hwnd: int) -> Optional[Tuple[int, int, int]]:
    """(width, height, refresh_hz) of the monitor hosting `hwnd`, or None."""
    try:
        gdi = ct.windll.gdi32
        _u32.MonitorFromWindow.restype = wintypes.HMONITOR
        _u32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        hmon = _u32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = MONITORINFOEXW()
        info.cbSize = ct.sizeof(MONITORINFOEXW)
        if not _u32.GetMonitorInfoW(hmon, ct.byref(info)):
            return None
        gdi.CreateDCW.restype = wintypes.HDC
        gdi.CreateDCW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, ct.c_void_p]
        gdi.GetDeviceCaps.argtypes = [wintypes.HDC, ct.c_int]
        gdi.DeleteDC.argtypes = [wintypes.HDC]
        hdc = gdi.CreateDCW(info.szDevice, None, None, None)
        if not hdc:
            return None
        try:
            # DESKTOPHORZRES=118, DESKTOPVERTRES=117, VREFRESH=116
            return gdi.GetDeviceCaps(hdc, 118), gdi.GetDeviceCaps(hdc, 117), gdi.GetDeviceCaps(hdc, 116)
        finally:
            gdi.DeleteDC(hdc)
    except Exception:
        return None

def _uptime_seconds() -> Optional[float]:
    try:
        k = ct.windll.kernel32
        k.GetTickCount64.restype = ct.c_ulonglong
        return k.GetTickCount64() / 1000.0
    except Exception:
        return None

# ===================== DRAW HELPERS =====================
def _hex_to_rgb(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

def _blend(c1: str, c2: str, t: float) -> str:
    """Mix c1 → c2 by t (0..1). Falls back to c2 on non-hex colors."""
    try:
        a = _hex_to_rgb(c1); b = _hex_to_rgb(c2)
    except Exception:
        return c2
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _fmt_gb(mb: Optional[float]) -> str:
    return "--" if mb is None else f"{mb / 1024.0:.1f} GB"

def _fmt_uptime(sec: Optional[float]) -> Tuple[str, str]:
    """(big text, small unit)"""
    if sec is None:
        return "--", "UP"
    m = int(sec // 60)
    d, m = divmod(m, 1440)
    h, m = divmod(m, 60)
    if d:
        return f"{d}d {h:02d}h", "UP"
    return f"{h:02d}:{m:02d}", "UP"

# ===================== DASHBOARD CANVAS =====================
class Dashboard(tk.Canvas):
    """Single canvas that draws the whole panel layout. Data comes from the owning DashboardTk."""

    GAUGE_SEGMENTS = 30
    GAUGE_START = 250.0   # degrees (math convention), arc runs clockwise from here
    GAUGE_SWEEP = 280.0

    def __init__(self, master: "DashboardTk"):
        super().__init__(master, bg=BG, highlightthickness=0)
        self.app = master
        self.u = 1.0
        self._fonts: Dict[tuple, tkfont.Font] = {}
        self._pending = None
        self.bind("<Configure>", lambda e: self.request_redraw(60))

    # ---------- infra ----------
    def font(self, px: float, bold: bool = True) -> tkfont.Font:
        key = (max(6, int(px)), bold)
        f = self._fonts.get(key)
        if f is None:
            f = tkfont.Font(root=self, family=self.app.display_font, size=-key[0],
                            weight="bold" if bold else "normal")
            self._fonts[key] = f
        return f

    def request_redraw(self, delay_ms: int = 0):
        if self._pending is not None:
            return
        self._pending = self.after(delay_ms, self.redraw) if delay_ms else self.after_idle(self.redraw)

    @staticmethod
    def _fit(text: str, font: tkfont.Font, maxw: float) -> str:
        if maxw <= 0:
            return ""
        if font.measure(text) <= maxw:
            return text
        while text and font.measure(text + "…") > maxw:
            text = text[:-1]
        return (text + "…") if text else ""

    # ---------- main layout ----------
    def redraw(self):
        self._pending = None
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        if W < 300 or H < 200:
            return
        a = self.app
        u = self.u = min(W / 1280.0, H / 720.0) * a.ui_scale
        m = a.outer_pad * u
        g = max(4.0, a.inner_gap * 2.8 * u)

        hdr = int(max(a.clock_pt * 1.2 * u, 30 * u) * 1.35)
        self._draw_header(m, 0, W - m, hdr)

        top_y0 = hdr
        avail = H - top_y0 - m
        top_h = (avail - g) * 0.54
        bot_y0 = top_y0 + top_h + g
        bot_y1 = H - m

        pw2 = (W - 2 * m - g) / 2.0
        self._panel_big(m, top_y0, m + pw2, top_y0 + top_h, "cpu")
        self._panel_big(m + pw2 + g, top_y0, W - m, top_y0 + top_h, "gpu")

        pw4 = (W - 2 * m - 3 * g) / 4.0
        xs = [m + i * (pw4 + g) for i in range(4)]
        self._panel_ram(xs[0], bot_y0, xs[0] + pw4, bot_y1)
        self._panel_fans(xs[1], bot_y0, xs[1] + pw4, bot_y1)
        self._panel_mobo(xs[2], bot_y0, xs[2] + pw4, bot_y1)
        self._panel_sys(xs[3], bot_y0, xs[3] + pw4, bot_y1)

    # ---------- header ----------
    def _draw_header(self, x0, y0, x1, y1):
        u = self.u
        a = self.app
        f = self.font(a.clock_pt * 1.2 * u)
        cy = (y0 + y1) / 2
        self.create_text(x0 + 14 * u, cy, anchor="w", text="", font=f, fill=a.clock_color, tags=("clock_time",))
        self.create_text((x0 + x1) / 2, cy, anchor="center", text="", font=f, fill=a.clock_color, tags=("clock_date",))
        bf = self.font(a.clock_pt * 0.95 * u)
        close_id = self.create_text(x1 - 6 * u, cy, anchor="e", text="✕", font=bf, fill=SUB, tags=("btn_close",))
        bx = self.bbox(close_id)
        gear_x = (bx[0] if bx else x1 - 50 * u) - 22 * u
        self.create_text(gear_x, cy, anchor="e", text="⚙", font=bf, fill=SUB, tags=("btn_settings",))
        for tag, cmd in (("btn_close", a.destroy), ("btn_settings", a._open_settings)):
            self.tag_bind(tag, "<Button-1>", lambda e, c=cmd: c())
            self.tag_bind(tag, "<Enter>", lambda e, t=tag: (self.itemconfigure(t, fill=FG), self.configure(cursor="hand2")))
            self.tag_bind(tag, "<Leave>", lambda e, t=tag: (self.itemconfigure(t, fill=SUB), self.configure(cursor="")))
        self.update_clock()

    def update_clock(self):
        now = datetime.now()
        col = self.app.clock_color
        self.itemconfigure("clock_time", text=f"{now:%H:%M:%S}", fill=col)
        self.itemconfigure("clock_date", text=f"{now:%d.%m.%Y}", fill=col)

    # ---------- panel frame ----------
    def _frame(self, x0, y0, x1, y1, title: str, subtitle: str, title_px: float,
               reserve_right: float = 0.0) -> Tuple[float, float, float, float, float]:
        """Draw the bracket-style frame. Returns (content x0, y0, x1, y1, header-right-area x0)."""
        u = self.u
        r = 16 * u
        thick = max(2, int(round(3.2 * u)))
        thin = max(1, int(round(1.4 * u)))
        hh = title_px * 1.18
        # corners + sides (thick), top/bottom edges (thin)
        self.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, style="arc", outline=FG, width=thick)
        self.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, style="arc", outline=FG, width=thick)
        self.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, style="arc", outline=FG, width=thick)
        self.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, style="arc", outline=FG, width=thick)
        self.create_line(x0, y0 + r, x0, y1 - r, fill=FG, width=thick)
        self.create_line(x1, y0 + r, x1, y1 - r, fill=FG, width=thick)
        self.create_line(x0 + r, y0, x1 - r, y0, fill=FG, width=thin)
        self.create_line(x0 + r, y1, x1 - r, y1, fill=FG, width=thin)
        # title
        tf = self.font(title_px)
        tx = x0 + r * 0.8
        self.create_text(tx, y0 + hh * 0.55, anchor="w", text=title, font=tf, fill=FG)
        tw = tf.measure(title)
        # stepped header underline
        yt = y0 + hh
        dd = hh * 0.30
        xa = tx + tw + 8 * u
        xb = xa + dd
        self.create_line(x0, yt, xa, yt, xb, yt - dd, x1, yt - dd, fill=FG, width=thin)
        dot = max(2.0, 2.6 * u)
        self.create_oval(xb - dot, yt - dd - dot, xb + dot, yt - dd + dot, fill=FG, outline="")
        # subtitle (sits on the upper step)
        right_x0 = x1 - r - reserve_right
        if subtitle:
            sf = self.font(max(10 * u, title_px * 0.34))
            sx = xb + 8 * u
            txt = self._fit(subtitle, sf, right_x0 - sx - 10 * u)
            self.create_text(sx, yt - dd - 3 * u, anchor="sw", text=txt, font=sf, fill=FG)
        pad = 12 * u
        return x0 + pad + 4 * u, yt + pad * 0.6, x1 - pad - 4 * u, y1 - pad, right_x0

    # ---------- primitives ----------
    def _hatch_bar(self, x0, y0, x1, y1, frac: Optional[float], color: str):
        bh = y1 - y0
        skew = bh * 0.9
        step = max(4.0, bh * 0.72)
        sw = step * 0.55
        fx = x0 + (x1 - x0) * _clamp01(frac) if frac is not None else x0 - 1
        x = x0
        while x + sw + skew <= x1:
            col = color if (x + (sw + skew) / 2) <= fx else DIM
            self.create_polygon(x, y1, x + sw, y1, x + sw + skew, y0, x + skew, y0, fill=col, outline="")
            x += step
        self.create_line(x0 + skew, y0 - max(1, bh * 0.25), x1, y0 - max(1, bh * 0.25), fill=DIM,
                         width=max(1, int(self.u)))

    def _row(self, x0, y0, x1, y1, label: str, value: str, frac: Optional[float], hot: bool = False):
        u = self.u
        h = y1 - y0
        f = self.font(min(h * 0.42, 26 * u))
        lx = x0 + 10 * u
        ty = y0 + h * 0.36
        col = HOT if hot else FG
        self.create_text(lx, ty, anchor="w", text=label, font=f, fill=FG)
        self.create_text(x1, ty, anchor="e", text=value, font=f, fill=col)
        by1 = y0 + h * 0.90
        by0 = by1 - max(4.0, h * 0.20)
        self._hatch_bar(lx, by0, x1, by1, frac, col)
        dot = max(2.0, 2.6 * u)
        self.create_oval(x0 - dot, by1 - dot, x0 + dot, by1 + dot, fill=FG, outline="")
        self.create_line(x0, by1, lx, by1, fill=FG, width=max(1, int(1.4 * u)))

    def _info_line(self, x0, y0, x1, y1, left: str, right: str = "", right_color: Optional[str] = None):
        """Text line with dot + underline (no bar)."""
        u = self.u
        h = y1 - y0
        f = self.font(min(h * 0.55, 24 * u))
        lx = x0 + 10 * u
        ty = y0 + h * 0.45
        self.create_text(lx, ty, anchor="w", text=left, font=f, fill=FG)
        if right:
            self.create_text(x1, ty, anchor="e", text=right, font=f, fill=right_color or FG)
        uy = y0 + h * 0.88
        dot = max(2.0, 2.6 * u)
        self.create_oval(x0 - dot, uy - dot, x0 + dot, uy + dot, fill=FG, outline="")
        self.create_line(x0, uy, lx, uy, x1, uy, fill=FG, width=max(1, int(1.2 * u)))

    def _histogram(self, x0, y0, x1, y1, values, vmax: float, color: str = None):
        u = self.u
        step = max(2, int(round(4 * u)))
        n = int((x1 - x0) // step)
        if n <= 0:
            return
        vals = list(values)[-n:]
        bw = max(1, int(round(step * 0.55)))
        x = x1 - len(vals) * step + step / 2
        hgt = y1 - y0
        for v in vals:
            v = _num(v)
            if v is not None:
                hh = max(1.0, _clamp01(v / max(1e-9, vmax)) * hgt)
                self.create_line(x, y1, x, y1 - hh, fill=color or FG, width=bw)
            x += step
        self.create_line(x0, y1 + max(1, u), x1, y1 + max(1, u), fill=DIM, width=max(1, int(u)))

    def _big_value(self, x0, y0, x1, y1, text: str, unit: str, color: str = None):
        """Large number with a small unit at its lower-right, centred in the box."""
        u = self.u
        px = (y1 - y0) * 0.85
        while px > 10:
            f = self.font(px)
            uf = self.font(px * 0.30)
            if f.measure(text) + uf.measure(unit) + 8 * u <= (x1 - x0):
                break
            px *= 0.9
        f = self.font(px); uf = self.font(px * 0.30)
        tw = f.measure(text); uw = uf.measure(unit)
        sx = (x0 + x1) / 2 - (tw + uw + 6 * u) / 2
        cy = (y0 + y1) / 2
        self.create_text(sx, cy, anchor="w", text=text, font=f, fill=color or FG)
        self.create_text(sx + tw + 6 * u, cy + px * 0.30, anchor="sw", text=unit, font=uf, fill=SUB)

    def _gauge(self, cx, cy, r, value, vmax: float, hot_ratio: float, label: str = "Temp."):
        u = self.u
        N = self.GAUGE_SEGMENTS
        seg = self.GAUGE_SWEEP / N
        gap = seg * 0.25
        tip = seg * 0.35
        r_out, r_in = r, r * 0.80
        r_mid = (r_out + r_in) / 2
        v = _num(value)
        frac = 0.0 if v is None else _clamp01(v / max(1e-9, vmax))
        hot = v is not None and v >= vmax * hot_ratio
        on = HOT if hot else FG
        nfill = int(round(frac * N))

        def P(rad, ang):
            t = math.radians(ang)
            return cx + rad * math.cos(t), cy - rad * math.sin(t)

        for i in range(N):
            a_s = self.GAUGE_START - i * seg
            a_e = a_s - seg + gap
            pts = (*P(r_out, a_s), *P(r_out, a_e), *P(r_mid, a_e - tip),
                   *P(r_in, a_e), *P(r_in, a_s), *P(r_mid, a_s - tip))
            self.create_polygon(pts, fill=on if i < nfill else DIM, outline="")

        ri = r_in - 7 * u
        end = self.GAUGE_START - self.GAUGE_SWEEP
        self.create_arc(cx - ri, cy - ri, cx + ri, cy + ri, start=end, extent=self.GAUGE_SWEEP,
                        style="arc", outline=DIM, width=max(1, int(1.5 * u)))

        # value + unit
        txt = "--" if v is None else f"{v:.0f}"
        npx = r * 0.62
        nf = self.font(npx)
        uf = self.font(npx * 0.34)
        tw = nf.measure(txt)
        nx = cx - r * 0.06
        self.create_text(nx, cy, anchor="center", text=txt, font=nf, fill=on)
        self.create_text(nx + tw / 2 + 4 * u, cy - npx * 0.22, anchor="w", text="°C", font=uf, fill=SUB)

        # "Temp." label in the bottom gap
        lf = self.font(max(10 * u, r * 0.15))
        lx, ly = P(r * 0.92, -62)
        self.create_text(lx, ly, anchor="w", text=label, font=lf, fill=FG)
        lw = lf.measure(label)
        uy = ly + r * 0.10
        dot = max(2.0, 2.6 * u)
        self.create_line(lx, uy, lx + lw, uy, fill=FG, width=max(1, int(1.4 * u)))
        self.create_oval(lx + lw - dot, uy - dot, lx + lw + dot, uy + dot, fill=FG, outline="")

    # ---------- panels ----------
    def _rows(self, x0, y0, x1, y1, rows: List[tuple], n_slots: int):
        rh = (y1 - y0) / max(1, n_slots)
        for i, row in enumerate(rows[:n_slots]):
            self._row(x0, y0 + i * rh, x1, y0 + (i + 1) * rh, *row)

    def _panel_big(self, x0, y0, x1, y1, key: str):
        a = self.app
        u = self.u
        d = a.data
        temp = d.get(f"{key}_temp")
        load = _num(d.get(f"{key}_usage"))
        fan = _num(d.get(f"fan_{key}_rpm"))
        vmax = a.vmax[key]
        tmax, tavg = a.temp_stats(key)
        hist_w = (x1 - x0) * 0.30
        cx0, cy0, cx1, cy1, hx0 = self._frame(x0, y0, x1, y1, key.upper(), a.labels.get(key, ""),
                                              title_px=52 * u, reserve_right=hist_w)
        # load history in the header
        hh = 52 * u * 1.18
        self._histogram(hx0, y0 + 8 * u, x1 - 16 * u, y0 + hh * 0.70 - 4 * u, a.hist[f"{key}_usage"], 100.0)

        gw = min(cy1 - cy0, (cx1 - cx0) * 0.46)
        r = gw / 2 * 0.96
        self._gauge(cx0 + gw / 2, (cy0 + cy1) / 2, r, temp, vmax, a.hot_ratio)

        def tfrac(t):
            return None if t is None else t / vmax

        def tval(t):
            return "--" if t is None else f"{t:.0f} °C"

        rows = [
            ("Load", "--%" if load is None else f"{load:.0f}%", None if load is None else load / 100.0),
            ("Fan", "-- RPM" if fan is None else f"{fan:.0f} RPM", a.fan_frac(f"fan_{key}_rpm", fan)),
            ("Max", tval(tmax), tfrac(tmax), tmax is not None and tmax >= vmax * a.hot_ratio),
            ("Avg", tval(tavg), tfrac(tavg), tavg is not None and tavg >= vmax * a.hot_ratio),
        ]
        self._rows(cx0 + gw + 16 * u, cy0, cx1, cy1, rows, 4)

    def _panel_ram(self, x0, y0, x1, y1):
        a = self.app
        u = self.u
        d = a.data
        pct = _num(d.get("ram_usage"))
        used = _num(d.get("ram_used_mb")); free = _num(d.get("ram_free_mb")); total = _num(d.get("ram_total_mb"))
        sub = "" if total is None else f"{round(total / 1024.0):.0f} GB"
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "RAM", sub, title_px=40 * u)
        hist_h = (cy1 - cy0) * 0.20
        rows = [
            ("Load", "--%" if pct is None else f"{pct:.0f}%", None if pct is None else pct / 100.0),
            ("Used", _fmt_gb(used), None if (used is None or not total) else used / total),
            ("Free", _fmt_gb(free), None if (free is None or not total) else free / total),
        ]
        self._rows(cx0, cy0, cx1, cy1 - hist_h, rows, 3)
        self._histogram(cx0 + 10 * u, cy1 - hist_h + 6 * u, cx1, cy1 - 2 * u, a.hist["ram_usage"], 100.0)

    def _panel_fans(self, x0, y0, x1, y1):
        a = self.app
        u = self.u
        fans = a.fan_list()
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "FANS", f"{len(fans)} sensors" if fans else "",
                                            title_px=40 * u)
        rows = [(name, "-- RPM" if v is None else f"{v:.0f} RPM", a.fan_frac(k, v)) for name, k, v in fans]
        if not rows:
            rows = [("Fan", "-- RPM", None)]
        self._rows(cx0, cy0, cx1, cy1, rows, 4)

    def _panel_mobo(self, x0, y0, x1, y1):
        a = self.app
        u = self.u
        temp = _num(a.data.get("pch_temp"))
        vmax = a.vmax["mobo"]
        tmax, tavg = a.temp_stats("mobo")
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "MOBO", a.labels.get("mobo", ""), title_px=40 * u)
        h = cy1 - cy0
        line_h = h * 0.20
        hist_h = h * 0.20
        mx = "--" if tmax is None else f"{tmax:.0f}°"
        av = "--" if tavg is None else f"{tavg:.0f}°"
        self._info_line(cx0, cy0, cx1, cy0 + line_h, f"Max {mx}", f"Avg {av}")
        hot = temp is not None and temp >= vmax * a.hot_ratio
        self._big_value(cx0, cy0 + line_h, cx1, cy1 - hist_h, "--" if temp is None else f"{temp:.0f}", "°C",
                        HOT if hot else FG)
        self._histogram(cx0 + 10 * u, cy1 - hist_h + 6 * u, cx1, cy1 - 2 * u, a.hist["pch_temp"], a.graph_max["mobo"])

    def _panel_sys(self, x0, y0, x1, y1):
        a = self.app
        u = self.u
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "SYS.", a.hostname, title_px=40 * u)
        h = cy1 - cy0
        line_h = h * 0.20
        di = a.display_info()
        res = "--" if not di else f"{di[0]} x {di[1]}"
        hz = "" if not di or di[2] <= 1 else f"{di[2]}Hz"
        self._info_line(cx0, cy0, cx1, cy0 + line_h, res, hz)
        big, unit = _fmt_uptime(_uptime_seconds())
        self._big_value(cx0, cy0 + line_h, cx1, cy1 - line_h, big, unit)
        status, col = ("ONLINE", FG) if a.connected else ("OFFLINE", OFFLINE)
        self._info_line(cx0, cy1 - line_h, cx1, cy1, "HWiNFO", status, col)


# ===================== UI APP =====================
class DashboardTk(tk.Tk):
    """Presentation-only Tkinter app. Use `post_snapshot(snapshot)` to feed data."""

    HIST_KEYS = ("cpu_temp", "pch_temp", "gpu_temp", "cpu_usage", "gpu_usage", "ram_usage")

    def __init__(self, cfg: dict):
        super().__init__()
        # `cfg` is shared with HwinfoReader (sensor_mapping) and updated in place by apply_config()
        self.cfg = cfg
        self.hist: Dict[str, Deque[Optional[float]]] = {}
        self.labels: Dict[str, str] = {"cpu": "CPU", "mobo": "Motherboard / PCH", "gpu": "GPU"}
        self._fan_peak: Dict[str, float] = {}
        self.data: Dict = {}
        self.connected = False
        self.hostname = os.environ.get("COMPUTERNAME") or platform.node()
        self._disp_cache: Tuple[float, Optional[Tuple[int, int, int]]] = (0.0, None)
        self.on_rescan = None  # set by the launcher: re-scan HWiNFO sensors after a mapping change
        self._read_cfg()

        # Window base
        self.configure(bg=BG)
        self.title("Dashboard")

        # Set window icon
        try:
            icon_path = os.path.join(SCRIPT_DIR, "hwmonitor.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except Exception:
            pass  # Icon not found or error loading, continue without it

        self._q: "queue.Queue[dict]" = queue.Queue()

        self.board = Dashboard(self)
        self.board.pack(expand=True, fill="both")

        # Hotkeys
        hk = cfg.get("hotkeys", {})
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind(f"<{hk.get('fullscreen_toggle', 'F11')}>", lambda e: self._toggle_fullscreen())

        # Start UI tickers
        self.after(200, self._tick_clock)
        self.after(100, self._drain_queue)

    def _read_cfg(self):
        """(Re)derive theme globals and UI parameters from self.cfg."""
        global BG, FG, SUB, GRID, TRACK, HOT, DIM, FONT_FAMILY
        cfg = self.cfg
        th = cfg.get("theme", {})
        BG = th.get("bg", BG); FG = th.get("fg", FG); SUB = th.get("sub", SUB)
        GRID = th.get("grid", GRID); TRACK = th.get("track", TRACK); HOT = th.get("hot", HOT)
        FONT_FAMILY = th.get("font_family", FONT_FAMILY)
        DIM = _blend(BG, FG, 0.22)
        self.ui_scale = float(th.get("ui_scale", 1.0))
        self.hot_ratio = float(th.get("hot_ratio", 0.85))
        wanted = th.get("display_font", "Bahnschrift")
        self.display_font = wanted if wanted in tkfont.families(self) else FONT_FAMILY

        fonts = cfg.get("fonts", {})
        rings = cfg.get("rings", {})
        graphs = cfg.get("graphs", {})
        layout = cfg.get("layout", {})
        poll = cfg.get("poll", {})
        fans_cfg = cfg.get("fans", {})

        self.clock_pt = float(fonts.get("clock_pt", 40))
        self.outer_pad = float(layout.get("outer_padding_px", 20))
        self.inner_gap = float(layout.get("inner_gap_px", 10))
        self.vmax = {k: float(rings.get(k, {}).get("vmax", dv)) for k, dv in (("cpu", 100), ("mobo", 95), ("gpu", 110))}
        self.graph_max = {k: float(graphs.get(k, {}).get("max_value", self.vmax[k] + 5)) for k in ("cpu", "mobo", "gpu")}
        self.fan_max_rpm = float(fans_cfg.get("max_rpm", 2000))
        self.pump_max_rpm = float(fans_cfg.get("pump_max_rpm", 3500))

        interval = max(0.1, float(poll.get("interval_sec", 1.0)))
        self.poll_ms = int(interval * 1000)
        n_hist = max(30, int(float(poll.get("history_seconds", 300)) / interval))
        for k in self.HIST_KEYS:
            old = self.hist.get(k)
            if old is None or old.maxlen != n_hist:
                self.hist[k] = deque(old or (), maxlen=n_hist)

        self._label_overrides = {k: str(v) for k, v in (cfg.get("labels", {}).get("override", {}) or {}).items() if v}
        self.labels.update(self._label_overrides)
        self.clock_color = FG if self.connected else OFFLINE

    def apply_config(self, cfg: dict, place: bool = False, rescan: bool = False):
        """Apply a new configuration live (used by the settings window)."""
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        _deep_merge(merged, json.loads(json.dumps(cfg)))
        self.cfg.clear()
        self.cfg.update(merged)
        self._read_cfg()
        self.configure(bg=BG)
        self.board.configure(bg=BG)
        self.board._fonts.clear()
        if place:
            self.place_window()
        if rescan and self.on_rescan:
            self.on_rescan()
        self.board.request_redraw()

    def place_window(self):
        """Position the window according to cfg.window (fullscreen on the chosen monitor, or windowed)."""
        win_cfg = self.cfg.get("window", {})
        monitor_cfg = (win_cfg.get("monitor") or {})
        if (win_cfg.get("mode") or "fullscreen").lower() == "fullscreen":
            mon = _pick_monitor(monitor_cfg)
            margins = monitor_cfg.get("safe_margins", {}) or {}
            l = int(margins.get("l", 0)); t = int(margins.get("t", 0)); r = int(margins.get("r", 0)); b = int(margins.get("b", 0))
            w = max(100, mon["w"] - (l + r)); h = max(100, mon["h"] - (t + b))
            self.overrideredirect(True)
            self.geometry(f"{w}x{h}+{mon['x'] + l}+{mon['y'] + t}")
        else:
            self.overrideredirect(False)
            self.geometry(win_cfg.get("geometry", "1280x720+100+100"))
        self.update_idletasks()

    # ---------- Public API ----------
    def set_labels(self, labels: Dict[str, str]):
        self.labels.update({k: v for k, v in labels.items() if k in ("cpu", "mobo", "gpu") and v})
        self.labels.update(self._label_overrides)
        self.board.request_redraw()

    def post_snapshot(self, snapshot: Dict):
        self._q.put(snapshot)

    def set_connection_status(self, connected: bool):
        """Update clock color + SYS panel based on connection status"""
        changed = connected != self.connected
        self.connected = connected
        self.clock_color = FG if connected else OFFLINE
        if changed:
            self.board.request_redraw()

    # ---------- Data helpers used by the canvas ----------
    def temp_stats(self, key: str) -> Tuple[Optional[float], Optional[float]]:
        hk = "pch_temp" if key == "mobo" else f"{key}_temp"
        vals = [v for v in (_num(x) for x in self.hist[hk]) if v is not None]
        if not vals:
            return None, None
        return max(vals), sum(vals) / len(vals)

    def fan_frac(self, key: str, rpm: Optional[float]) -> Optional[float]:
        if rpm is None:
            return None
        base = self.pump_max_rpm if "pump" in key.lower() else self.fan_max_rpm
        peak = max(base, self._fan_peak.get(key, 0.0))
        return rpm / peak if peak > 0 else None

    def fan_list(self) -> List[Tuple[str, str, Optional[float]]]:
        """[(display name, key, rpm)] — CPU/GPU/Pump first, then system fans."""
        d = self.data
        fixed = [("CPU", "fan_cpu_rpm", _num(d.get("fan_cpu_rpm"))),
                 ("GPU", "fan_gpu_rpm", _num(d.get("fan_gpu_rpm"))),
                 ("Pump", "fan_pump_rpm", _num(d.get("fan_pump_rpm")))]
        sys_fans = [(str(n), f"sys:{n}", _num(v)) for n, v in ((d.get("fans") or {}).get("sys") or {}).items()]
        out = [f for f in fixed if f[2] is not None] + sys_fans
        return out or fixed

    def display_info(self) -> Optional[Tuple[int, int, int]]:
        t, info = self._disp_cache
        if info is None or time.time() - t > 10:
            info = _display_info(self.winfo_id())
            self._disp_cache = (time.time(), info)
        return info

    # ---------- Internals ----------
    def _tick_clock(self):
        self.board.update_clock()
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
        if drained:
            self.board.request_redraw()
        self.after(100, self._drain_queue)

    def _apply_snapshot(self, snap: Dict):
        if isinstance(snap.get('labels'), dict):
            self.labels.update({k: v for k, v in snap['labels'].items() if k in ("cpu", "mobo", "gpu") and v})
            self.labels.update(self._label_overrides)
        self.data = snap
        for k in self.hist:
            self.hist[k].append(_num(snap.get(k)))
        for name, key, rpm in self.fan_list():
            if rpm is not None:
                self._fan_peak[key] = max(self._fan_peak.get(key, 0.0), rpm)

    def _open_settings(self):
        """Open settings window"""
        try:
            from settings_window import SettingsWindow
            SettingsWindow(self)
        except Exception as e:
            print(f"Error opening settings: {e}")


    def _toggle_fullscreen(self):
        cur = self.overrideredirect()
        self.overrideredirect(not cur)
        self.update_idletasks()

# ===================== LAUNCHER (reads config + hwreader) =====================
if __name__ == "__main__":
    cfg = load_config("hwconfig.json")

    app = DashboardTk(cfg)
    app.place_window()

    # Data source (hwreader) with auto-reconnect
    reconnect_ms = 60000  # 1 minute

    from hwreader import HwinfoReader

    # State container to avoid nonlocal issues in module scope
    state = {
        'rdr': None,
        'is_connected': False,
        'reconnect_timer_id': None
    }

    def try_connect():
        """Attempt to connect to HWiNFO shared memory"""
        try:
            if state['rdr'] is None:
                state['rdr'] = HwinfoReader(cfg)  # Pass config for sensor mapping
            state['rdr'].open()
            state['rdr'].scan()
            state['is_connected'] = True
            try:
                app.set_labels(state['rdr'].labels())
            except Exception:
                pass
            app.set_connection_status(True)  # White when connected
            return True
        except Exception as e:
            state['is_connected'] = False
            app.set_connection_status(False)  # Orange when disconnected
            return False

    def tick():
        """Main polling loop for reading sensor data"""
        if not state['is_connected']:
            # Not connected, skip this tick
            app.after(app.poll_ms, tick)
            return

        try:
            snap = state['rdr'].read_snapshot()
            app.post_snapshot(snap)
            # Connection working fine
        except Exception as e:
            # Connection lost
            state['is_connected'] = False
            app.set_connection_status(False)  # Orange when connection lost
            # Start reconnection timer if not already running
            if state['reconnect_timer_id'] is None:
                start_reconnect_timer()
        finally:
            app.after(app.poll_ms, tick)

    def reconnect_attempt():
        """Periodic reconnection attempt (every minute)"""
        if state['is_connected']:
            # Already connected, stop the timer
            state['reconnect_timer_id'] = None
            return

        # Try to reconnect
        if try_connect():
            # Success! Stop the reconnection timer
            state['reconnect_timer_id'] = None
        else:
            # Failed, schedule next attempt
            state['reconnect_timer_id'] = app.after(reconnect_ms, reconnect_attempt)

    def start_reconnect_timer():
        """Start the reconnection timer"""
        if state['reconnect_timer_id'] is None:
            state['reconnect_timer_id'] = app.after(reconnect_ms, reconnect_attempt)

    def on_close():
        """Cleanup on window close"""
        # Cancel reconnect timer if running
        if state['reconnect_timer_id'] is not None:
            try:
                app.after_cancel(state['reconnect_timer_id'])
            except Exception:
                pass
        # Close reader
        try:
            if state['rdr'] is not None:
                state['rdr'].close()
        except Exception:
            pass
        finally:
            app.destroy()

    # Initial connection attempt
    if not try_connect():
        # Failed initial connection, start reconnection timer
        start_reconnect_timer()

    # Bind hotkeys
    hk = cfg.get("hotkeys", {})
    rescan_key = hk.get("rescan_hwinfo", "R")

    def manual_rescan(e=None):
        """Manual rescan triggered by hotkey (or by the settings window after a sensor change)"""
        if state['is_connected'] and state['rdr'] is not None:
            try:
                state['rdr'].scan()
                app.set_labels(state['rdr'].labels())
            except Exception:
                pass

    app.bind(f"<{rescan_key}>", manual_rescan)
    app.on_rescan = manual_rescan
    app.protocol("WM_DELETE_WINDOW", on_close)

    # Start main polling loop
    tick()

    app.mainloop()
