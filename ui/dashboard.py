"""
Dashboard window (Tkinter).
- DashboardApp: the Tk root; owns config, theme, SensorState; fed via post_snapshot().
- Dashboard:    a single Canvas that draws the panel layout
  * top row:    CPU / GPU panels (segmented temperature gauge + Load / Fan / Max / Avg rows)
  * bottom row: RAM / FANS / MOBO / SYS panels
Presentation only: no HWiNFO access here (see datasource.HwinfoPoller).
"""

from __future__ import annotations

import math
import os
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import config
import winsys
from model import COMPONENTS, SensorState, Snapshot, num
from theme import Theme, resolve_font


# ===================== formatting =====================
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _fmt_gb(mb: Optional[float]) -> str:
    return "--" if mb is None else f"{mb / 1024.0:.1f} GB"

def _fmt_duration(sec: Optional[float]) -> str:
    if sec is None:
        return "--"
    m = int(sec // 60)
    d, m = divmod(m, 1440)
    h, m = divmod(m, 60)
    return f"{d}d {h:02d}h" if d else f"{h:02d}:{m:02d}"


# ===================== canvas =====================
class Dashboard(tk.Canvas):
    """Single canvas that draws the whole panel layout. Data comes from the owning DashboardApp."""

    GAUGE_SEGMENTS = 30
    GAUGE_START = 250.0   # degrees (math convention), arc runs clockwise from here
    GAUGE_SWEEP = 280.0

    def __init__(self, app: "DashboardApp"):
        super().__init__(app, bg=app.theme.bg, highlightthickness=0)
        self.app = app
        self.t = app.theme
        self.u = 1.0
        self._fonts: Dict[tuple, tkfont.Font] = {}
        self._pending = None
        self.bind("<Configure>", lambda e: self.request_redraw(60))

    # ---------- infra ----------
    def set_theme(self, theme: Theme):
        self.t = theme
        self._fonts.clear()
        self.configure(bg=theme.bg)

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
        u, t, a = self.u, self.t, self.app
        f = self.font(a.clock_pt * 1.2 * u)
        cy = (y0 + y1) / 2
        self.create_text(x0 + 14 * u, cy, anchor="w", text="", font=f, tags=("clock_time",))
        self.create_text((x0 + x1) / 2, cy, anchor="center", text="", font=f, tags=("clock_date",))
        bf = self.font(a.clock_pt * 0.95 * u)
        close_id = self.create_text(x1 - 6 * u, cy, anchor="e", text="✕", font=bf, fill=t.sub, tags=("btn_close",))
        bx = self.bbox(close_id)
        gear_x = (bx[0] if bx else x1 - 50 * u) - 22 * u
        self.create_text(gear_x, cy, anchor="e", text="⚙", font=bf, fill=t.sub, tags=("btn_settings",))
        for tag, cmd in (("btn_close", a.close), ("btn_settings", a.open_settings)):
            self.tag_bind(tag, "<Button-1>", lambda e, c=cmd: c())
            self.tag_bind(tag, "<Enter>", lambda e, tg=tag: (self.itemconfigure(tg, fill=self.t.fg), self.configure(cursor="hand2")))
            self.tag_bind(tag, "<Leave>", lambda e, tg=tag: (self.itemconfigure(tg, fill=self.t.sub), self.configure(cursor="")))
        self.update_clock()

    def update_clock(self):
        now = datetime.now()
        col = self.t.fg if self.app.connected else self.t.offline
        self.itemconfigure("clock_time", text=f"{now:%H:%M:%S}", fill=col)
        self.itemconfigure("clock_date", text=f"{now:%d.%m.%Y}", fill=col)

    # ---------- panel frame ----------
    def _frame(self, x0, y0, x1, y1, title: str, subtitle: str, title_px: float,
               reserve_right: float = 0.0) -> Tuple[float, float, float, float, float]:
        """Draw the bracket-style frame. Returns (content x0, y0, x1, y1, header-right-area x0)."""
        u, fg = self.u, self.t.fg
        r = 16 * u
        thick = max(2, int(round(3.2 * u)))
        thin = max(1, int(round(1.4 * u)))
        hh = title_px * 1.18
        # corners + sides (thick), top/bottom edges (thin)
        self.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, style="arc", outline=fg, width=thick)
        self.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, style="arc", outline=fg, width=thick)
        self.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, style="arc", outline=fg, width=thick)
        self.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, style="arc", outline=fg, width=thick)
        self.create_line(x0, y0 + r, x0, y1 - r, fill=fg, width=thick)
        self.create_line(x1, y0 + r, x1, y1 - r, fill=fg, width=thick)
        self.create_line(x0 + r, y0, x1 - r, y0, fill=fg, width=thin)
        self.create_line(x0 + r, y1, x1 - r, y1, fill=fg, width=thin)
        # title
        tf = self.font(title_px)
        tx = x0 + r * 0.8
        self.create_text(tx, y0 + hh * 0.55, anchor="w", text=title, font=tf, fill=fg)
        tw = tf.measure(title)
        # stepped header underline
        yt = y0 + hh
        dd = hh * 0.30
        xa = tx + tw + 8 * u
        xb = xa + dd
        self.create_line(x0, yt, xa, yt, xb, yt - dd, x1, yt - dd, fill=fg, width=thin)
        dot = max(2.0, 2.6 * u)
        self.create_oval(xb - dot, yt - dd - dot, xb + dot, yt - dd + dot, fill=fg, outline="")
        # subtitle (sits on the upper step)
        right_x0 = x1 - r - reserve_right
        if subtitle:
            sf = self.font(max(10 * u, title_px * 0.34))
            sx = xb + 8 * u
            txt = self._fit(subtitle, sf, right_x0 - sx - 10 * u)
            self.create_text(sx, yt - dd - 3 * u, anchor="sw", text=txt, font=sf, fill=fg)
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
            col = color if (x + (sw + skew) / 2) <= fx else self.t.dim
            self.create_polygon(x, y1, x + sw, y1, x + sw + skew, y0, x + skew, y0, fill=col, outline="")
            x += step
        self.create_line(x0 + skew, y0 - max(1, bh * 0.25), x1, y0 - max(1, bh * 0.25), fill=self.t.dim,
                         width=max(1, int(self.u)))

    def _row(self, x0, y0, x1, y1, label: str, value: str, frac: Optional[float], hot: bool = False):
        u, t = self.u, self.t
        h = y1 - y0
        f = self.font(min(h * 0.42, 26 * u))
        lx = x0 + 10 * u
        ty = y0 + h * 0.36
        col = t.hot if hot else t.fg
        self.create_text(lx, ty, anchor="w", text=label, font=f, fill=t.fg)
        self.create_text(x1, ty, anchor="e", text=value, font=f, fill=col)
        by1 = y0 + h * 0.90
        by0 = by1 - max(4.0, h * 0.20)
        self._hatch_bar(lx, by0, x1, by1, frac, col)
        dot = max(2.0, 2.6 * u)
        self.create_oval(x0 - dot, by1 - dot, x0 + dot, by1 + dot, fill=t.fg, outline="")
        self.create_line(x0, by1, lx, by1, fill=t.fg, width=max(1, int(1.4 * u)))

    def _info_line(self, x0, y0, x1, y1, left: str, right: str = "", right_color: Optional[str] = None):
        """Text line with dot + underline (no bar)."""
        u, t = self.u, self.t
        h = y1 - y0
        f = self.font(min(h * 0.55, 24 * u))
        lx = x0 + 10 * u
        ty = y0 + h * 0.45
        self.create_text(lx, ty, anchor="w", text=left, font=f, fill=t.fg)
        if right:
            self.create_text(x1, ty, anchor="e", text=right, font=f, fill=right_color or t.fg)
        uy = y0 + h * 0.88
        dot = max(2.0, 2.6 * u)
        self.create_oval(x0 - dot, uy - dot, x0 + dot, uy + dot, fill=t.fg, outline="")
        self.create_line(x0, uy, lx, uy, x1, uy, fill=t.fg, width=max(1, int(1.2 * u)))

    def _histogram(self, x0, y0, x1, y1, values, vmax: float):
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
            if v is not None:
                hh = max(1.0, _clamp01(v / max(1e-9, vmax)) * hgt)
                self.create_line(x, y1, x, y1 - hh, fill=self.t.fg, width=bw)
            x += step
        self.create_line(x0, y1 + max(1, u), x1, y1 + max(1, u), fill=self.t.dim, width=max(1, int(u)))

    def _big_value(self, x0, y0, x1, y1, text: str, unit: str, color: Optional[str] = None):
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
        tw = f.measure(text)
        sx = (x0 + x1) / 2 - (tw + uf.measure(unit) + 6 * u) / 2
        cy = (y0 + y1) / 2
        self.create_text(sx, cy, anchor="w", text=text, font=f, fill=color or self.t.fg)
        self.create_text(sx + tw + 6 * u, cy + px * 0.30, anchor="sw", text=unit, font=uf, fill=self.t.sub)

    def _gauge(self, cx, cy, r, value: Optional[float], vmax: float, hot_ratio: float, label: str = "Temp."):
        u, t = self.u, self.t
        N = self.GAUGE_SEGMENTS
        seg = self.GAUGE_SWEEP / N
        gap = seg * 0.25
        tip = seg * 0.35
        r_out, r_in = r, r * 0.80
        r_mid = (r_out + r_in) / 2
        frac = 0.0 if value is None else _clamp01(value / max(1e-9, vmax))
        hot = value is not None and value >= vmax * hot_ratio
        on = t.hot if hot else t.fg
        nfill = int(round(frac * N))

        def P(rad, ang):
            a = math.radians(ang)
            return cx + rad * math.cos(a), cy - rad * math.sin(a)

        for i in range(N):
            a_s = self.GAUGE_START - i * seg
            a_e = a_s - seg + gap
            pts = (*P(r_out, a_s), *P(r_out, a_e), *P(r_mid, a_e - tip),
                   *P(r_in, a_e), *P(r_in, a_s), *P(r_mid, a_s - tip))
            self.create_polygon(pts, fill=on if i < nfill else t.dim, outline="")

        ri = r_in - 7 * u
        end = self.GAUGE_START - self.GAUGE_SWEEP
        self.create_arc(cx - ri, cy - ri, cx + ri, cy + ri, start=end, extent=self.GAUGE_SWEEP,
                        style="arc", outline=t.dim, width=max(1, int(1.5 * u)))

        # value + unit
        txt = "--" if value is None else f"{value:.0f}"
        npx = r * 0.62
        nf = self.font(npx)
        uf = self.font(npx * 0.34)
        tw = nf.measure(txt)
        nx = cx - r * 0.06
        self.create_text(nx, cy, anchor="center", text=txt, font=nf, fill=on)
        self.create_text(nx + tw / 2 + 4 * u, cy - npx * 0.22, anchor="w", text="°C", font=uf, fill=t.sub)

        # "Temp." label in the bottom gap
        lf = self.font(max(10 * u, r * 0.15))
        lx, ly = P(r * 0.92, -62)
        self.create_text(lx, ly, anchor="w", text=label, font=lf, fill=t.fg)
        lw = lf.measure(label)
        uy = ly + r * 0.10
        dot = max(2.0, 2.6 * u)
        self.create_line(lx, uy, lx + lw, uy, fill=t.fg, width=max(1, int(1.4 * u)))
        self.create_oval(lx + lw - dot, uy - dot, lx + lw + dot, uy + dot, fill=t.fg, outline="")

    # ---------- panels ----------
    def _rows(self, x0, y0, x1, y1, rows: List[tuple], n_slots: int):
        rh = (y1 - y0) / max(1, n_slots)
        for i, row in enumerate(rows[:n_slots]):
            self._row(x0, y0 + i * rh, x1, y0 + (i + 1) * rh, *row)

    def _panel_big(self, x0, y0, x1, y1, comp: str):
        a, u = self.app, self.u
        s = a.state.latest
        temp = s.temp(comp)
        load = num(getattr(s, f"{comp}_usage"))
        fan = num(getattr(s, f"fan_{comp}_rpm"))
        vmax = a.vmax[comp]
        tmax, tavg = a.state.temp_stats(comp)
        hist_w = (x1 - x0) * 0.30
        cx0, cy0, cx1, cy1, hx0 = self._frame(x0, y0, x1, y1, comp.upper(), a.label(comp),
                                              title_px=52 * u, reserve_right=hist_w)
        # load history in the header
        hh = 52 * u * 1.18
        self._histogram(hx0, y0 + 8 * u, x1 - 16 * u, y0 + hh * 0.70 - 4 * u, a.state.history[f"{comp}_usage"], 100.0)

        gw = min(cy1 - cy0, (cx1 - cx0) * 0.46)
        self._gauge(cx0 + gw / 2, (cy0 + cy1) / 2, gw / 2 * 0.96, temp, vmax, a.hot_ratio)

        def trow(name, tv):
            return (name, "--" if tv is None else f"{tv:.0f} °C", None if tv is None else tv / vmax,
                    tv is not None and tv >= vmax * a.hot_ratio)

        fan_reading = next((f for f in a.state.fans() if f.key == comp), None)
        rows = [
            ("Load", "--%" if load is None else f"{load:.0f}%", None if load is None else load / 100.0),
            ("Fan", "-- RPM" if fan is None else f"{fan:.0f} RPM",
             a.state.fan_fraction(fan_reading, a.fan_max_rpm) if fan_reading else None),
            trow("Max", tmax),
            trow("Avg", tavg),
        ]
        self._rows(cx0 + gw + 16 * u, cy0, cx1, cy1, rows, 4)

    def _panel_ram(self, x0, y0, x1, y1):
        a, u = self.app, self.u
        s = a.state.latest
        pct = num(s.ram_usage)
        used, free, total = num(s.ram_used_mb), num(s.ram_free_mb), num(s.ram_total_mb)
        sub = "" if total is None else f"{round(total / 1024.0):.0f} GB"
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "RAM", sub, title_px=40 * u)
        hist_h = (cy1 - cy0) * 0.20
        rows = [
            ("Load", "--%" if pct is None else f"{pct:.0f}%", None if pct is None else pct / 100.0),
            ("Used", _fmt_gb(used), None if (used is None or not total) else used / total),
            ("Free", _fmt_gb(free), None if (free is None or not total) else free / total),
        ]
        self._rows(cx0, cy0, cx1, cy1 - hist_h, rows, 3)
        self._histogram(cx0 + 10 * u, cy1 - hist_h + 6 * u, cx1, cy1 - 2 * u, a.state.history["ram_usage"], 100.0)

    def _panel_fans(self, x0, y0, x1, y1):
        a, u = self.app, self.u
        fans = a.state.fans()
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "FANS", f"{len(fans)} sensors" if fans else "",
                                            title_px=40 * u)
        rows = [(f.name, "-- RPM" if f.rpm is None else f"{f.rpm:.0f} RPM",
                 a.state.fan_fraction(f, a.pump_max_rpm if f.is_pump else a.fan_max_rpm)) for f in fans]
        self._rows(cx0, cy0, cx1, cy1, rows or [("Fan", "-- RPM", None)], 4)

    def _panel_mobo(self, x0, y0, x1, y1):
        a, u, t = self.app, self.u, self.t
        temp = a.state.latest.temp("mobo")
        vmax = a.vmax["mobo"]
        tmax, tavg = a.state.temp_stats("mobo")
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "MOBO", a.label("mobo"), title_px=40 * u)
        h = cy1 - cy0
        line_h = h * 0.20
        hist_h = h * 0.20
        mx = "--" if tmax is None else f"{tmax:.0f}°"
        av = "--" if tavg is None else f"{tavg:.0f}°"
        self._info_line(cx0, cy0, cx1, cy0 + line_h, f"Max {mx}", f"Avg {av}")
        hot = temp is not None and temp >= vmax * a.hot_ratio
        self._big_value(cx0, cy0 + line_h, cx1, cy1 - hist_h, "--" if temp is None else f"{temp:.0f}", "°C",
                        t.hot if hot else t.fg)
        self._histogram(cx0 + 10 * u, cy1 - hist_h + 6 * u, cx1, cy1 - 2 * u, a.state.history["mobo_temp"],
                        a.mobo_graph_max)

    def _panel_sys(self, x0, y0, x1, y1):
        a, u, t = self.app, self.u, self.t
        cx0, cy0, cx1, cy1, _ = self._frame(x0, y0, x1, y1, "SYS.", a.hostname, title_px=40 * u)
        h = cy1 - cy0
        line_h = h * 0.20
        di = a.display_info()
        res = "--" if not di else f"{di[0]} x {di[1]}"
        hz = "" if not di or di[2] <= 1 else f"{di[2]}Hz"
        self._info_line(cx0, cy0, cx1, cy0 + line_h, res, hz)

        # Big value: how long HWiNFO has been running (free version stops sharing data after 12h);
        # falls back to Windows uptime when HWiNFO is not running.
        hw_up = a.hwinfo_uptime()
        if hw_up is not None:
            warn = hw_up >= a.hwinfo_warn_hours * 3600
            self._big_value(cx0, cy0 + line_h, cx1, cy1 - line_h, _fmt_duration(hw_up), "UP",
                            t.hot if warn else t.fg)
        else:
            warn = False
            self._big_value(cx0, cy0 + line_h, cx1, cy1 - line_h, _fmt_duration(winsys.uptime_seconds()), "PC")

        if warn:
            status, col = "RESTART", t.hot
        elif not a.connected:
            status, col = "OFFLINE", t.offline
        else:
            status, col = "ONLINE", t.fg
        self._info_line(cx0, cy1 - line_h, cx1, cy1, "HWiNFO", status, col)


# ===================== app =====================
class DashboardApp(tk.Tk):
    """Tk root. Feed it with post_snapshot(Snapshot); change config live with apply_config()."""

    def __init__(self, cfg: dict):
        super().__init__()
        # `cfg` is shared with the poller/reader and updated in place by apply_config()
        self.cfg = cfg
        self.state = SensorState()
        self.connected = False
        self._ever_connected = False
        self._hw_session_start: Optional[float] = None   # set on reconnect → resets the UP timer
        self.hostname = winsys.hostname()
        self.reader_labels: Dict[str, str] = {}
        self.on_rescan: Optional[Callable[[], None]] = None   # set by main: re-scan sensors
        self._close_handlers: List[Callable[[], None]] = []
        self._cache: Dict[str, Tuple[float, object]] = {}
        self._read_cfg()

        self.configure(bg=self.theme.bg)
        self.title("Dashboard")
        try:
            if os.path.exists(config.ICON_FILE):
                self.iconbitmap(config.ICON_FILE)
        except Exception:
            pass  # Icon not found or error loading, continue without it

        self.board = Dashboard(self)
        self.board.pack(expand=True, fill="both")

        hk = cfg.get("hotkeys", {})
        self.bind("<Escape>", lambda e: self.close())
        self.bind(f"<{hk.get('fullscreen_toggle', 'F11')}>", lambda e: self.toggle_fullscreen())
        self.bind(f"<{hk.get('rescan_hwinfo', 'R')}>", lambda e: self.on_rescan and self.on_rescan())
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._tick_clock()

    # ---------- config ----------
    def _read_cfg(self):
        """Derive theme and UI parameters from self.cfg."""
        cfg = self.cfg
        th = cfg.get("theme", {})
        g = lambda path, dv: config.get_path(cfg, path, dv)
        self.theme = Theme.from_config(cfg)
        self.ui_scale = float(th.get("ui_scale", 1.0))
        self.hot_ratio = float(th.get("hot_ratio", 0.85))
        self.display_font = resolve_font(self, th.get("display_font", "Bahnschrift"), th.get("font_family", "Consolas"))
        self.clock_pt = float(g("fonts.clock_pt", 40))
        self.outer_pad = float(g("layout.outer_padding_px", 20))
        self.inner_gap = float(g("layout.inner_gap_px", 10))
        self.vmax = {c: float(g(f"rings.{c}.vmax", dv)) for c, dv in (("cpu", 100), ("gpu", 110), ("mobo", 95))}
        self.mobo_graph_max = float(g("graphs.mobo.max_value", self.vmax["mobo"] + 5))
        self.fan_max_rpm = float(g("fans.max_rpm", 2000))
        self.pump_max_rpm = float(g("fans.pump_max_rpm", 3500))
        self.hwinfo_warn_hours = float(g("hwinfo.warn_hours", 11.5))
        self.hwinfo_processes = list(g("hwinfo.process_names", ["HWiNFO64.exe", "HWiNFO32.exe", "HWiNFO.exe"]))
        interval = max(0.1, float(g("poll.interval_sec", 1.0)))
        self.state.resize(max(30, int(float(g("poll.history_seconds", 300)) / interval)))
        self.label_overrides = {k: str(v) for k, v in (g("labels.override", {}) or {}).items() if v}

    def apply_config(self, cfg: dict, place: bool = False, rescan: bool = False):
        """Apply a new configuration live (used by the settings window)."""
        merged = config.with_defaults(cfg)
        self.cfg.clear()
        self.cfg.update(merged)
        self._read_cfg()
        self.configure(bg=self.theme.bg)
        self.board.set_theme(self.theme)
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
            mon = winsys.pick_monitor(monitor_cfg)
            mg = monitor_cfg.get("safe_margins", {}) or {}
            l, t, r, b = (int(mg.get(k, 0)) for k in ("l", "t", "r", "b"))
            w = max(100, mon["w"] - (l + r)); h = max(100, mon["h"] - (t + b))
            self.overrideredirect(True)
            self.geometry(f"{w}x{h}+{mon['x'] + l}+{mon['y'] + t}")
        else:
            self.overrideredirect(False)
            self.geometry(win_cfg.get("geometry", "1280x720+100+100"))
        self.update_idletasks()

    # ---------- data in ----------
    def post_snapshot(self, snap: Snapshot):
        self.state.update(snap)
        if snap.labels:
            self.reader_labels.update({k: v for k, v in snap.labels.items() if v})
        self.board.request_redraw()

    def set_labels(self, labels: Dict[str, str]):
        self.reader_labels.update({k: v for k, v in labels.items() if v})
        self.board.request_redraw()

    def set_connection_status(self, connected: bool):
        if connected != self.connected:
            if connected:
                # Reconnected (e.g. HWiNFO free resumed sharing after the 12h stop): restart the UP timer
                # from now, unless this is the first connection to a HWiNFO that is still within its window.
                up = self.hwinfo_uptime()
                if self._ever_connected or (up is not None and up >= self.hwinfo_warn_hours * 3600):
                    self._hw_session_start = time.time()
                self._ever_connected = True
            self.connected = connected
            self.board.request_redraw()

    # ---------- helpers for the canvas ----------
    def label(self, comp: str) -> str:
        default = {"cpu": "CPU", "gpu": "GPU", "mobo": "Motherboard / PCH"}[comp]
        return self.label_overrides.get(comp) or self.reader_labels.get(comp) or default

    def _cached(self, key: str, ttl: float, fn):
        t, val = self._cache.get(key, (0.0, None))
        if time.time() - t > ttl:
            val = fn()
            self._cache[key] = (time.time(), val)
        return val

    def display_info(self) -> Optional[Tuple[int, int, int]]:
        return self._cached("display", 10, lambda: winsys.display_info(self.winfo_id()))

    def hwinfo_uptime(self) -> Optional[float]:
        start = self._cached("hwinfo_start", 10, lambda: winsys.process_start_time(self.hwinfo_processes))
        if start is None:
            return None
        if self._hw_session_start is not None:
            start = max(start, self._hw_session_start)  # a newer HWiNFO process still wins
        return max(0.0, time.time() - start)

    # ---------- window ----------
    def add_close_handler(self, fn: Callable[[], None]):
        self._close_handlers.append(fn)

    def close(self):
        for fn in self._close_handlers:
            try:
                fn()
            except Exception:
                pass
        self.destroy()

    def open_settings(self):
        try:
            from ui.settings import SettingsWindow
            SettingsWindow(self)
        except Exception as e:
            print(f"Error opening settings: {e}")

    def toggle_fullscreen(self):
        self.overrideredirect(not self.overrideredirect())
        self.update_idletasks()

    def _tick_clock(self):
        self.board.update_clock()
        self.after(200, self._tick_clock)
