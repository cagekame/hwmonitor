"""
Settings window for HW Monitor Dashboard.
- Same visual language as the dashboard (theme colors, display font, hatched bars).
- Every change is previewed live on the dashboard; Save writes hwconfig.json, Cancel reverts.
- Window / monitor placement and sensor mapping are applied on Save.
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, colorchooser
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from UI_Dashboard import DashboardTk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "hwconfig.json")

PRESETS: Dict[str, Dict[str, str]] = {
    "Dark":  {"bg": "#0b0e13", "fg": "#e5eef9", "sub": "#9fb0c6", "hot": "#ff5a4f"},
    "Light": {"bg": "#f3f4f6", "fg": "#111318", "sub": "#5b6472", "hot": "#d93025"},
    "Neon":  {"bg": "#06070c", "fg": "#3ee8ff", "sub": "#8b7dff", "hot": "#ff3fd2"},
    "Amber": {"bg": "#0d0a05", "fg": "#ffb547", "sub": "#a88a5c", "hot": "#ff4b3a"},
}
AUTO_SENSOR = "Auto-detect"


# ===================== helpers =====================
def _blend(c1: str, c2: str, t: float) -> str:
    try:
        a = [int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
        b = [int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    except Exception:
        return c2
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def _valid_hex(c: str) -> bool:
    c = (c or "").strip()
    if len(c) != 7 or not c.startswith("#"):
        return False
    try:
        int(c[1:], 16)
        return True
    except ValueError:
        return False

def _get(d: dict, path: str, default=None):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return default
        d = d[part]
    return d

def _set(d: dict, path: str, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(d.get(part), dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


class Palette:
    def __init__(self, bg: str, fg: str, sub: str, hot: str):
        self.bg, self.fg, self.sub, self.hot = bg, fg, sub, hot
        self.panel = _blend(bg, fg, 0.04)
        self.field = _blend(bg, fg, 0.09)
        self.hover = _blend(bg, fg, 0.14)
        self.line = _blend(bg, fg, 0.30)
        self.dim = _blend(bg, fg, 0.22)


# ===================== custom widgets =====================
class HatchSlider(tk.Canvas):
    """Slider drawn like the dashboard bars. Click or drag to set; arrow keys nudge."""

    def __init__(self, parent, pal: Palette, font: tkfont.Font, var: tk.DoubleVar,
                 from_: float, to: float, step: float = 1, fmt: Callable[[float], str] = None,
                 on_change: Callable[[], None] = None):
        super().__init__(parent, height=30, bg=pal.panel, highlightthickness=0, cursor="hand2", takefocus=1)
        self.pal, self.f, self.var = pal, font, var
        self.from_, self.to, self.step = float(from_), float(to), float(step)
        self.fmt = fmt or (lambda v: f"{v:g}")
        self.on_change = on_change
        self._val_w = 96
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Button-1>", self._from_mouse)
        self.bind("<B1-Motion>", self._from_mouse)
        self.bind("<Left>", lambda e: self._nudge(-1))
        self.bind("<Right>", lambda e: self._nudge(1))
        var.trace_add("write", lambda *a: self.draw())

    def _bounds(self):
        w = self.winfo_width()
        return 4, max(10, w - self._val_w - 12)

    def _value(self) -> float:
        try:
            return float(self.var.get())
        except (tk.TclError, ValueError):
            return self.from_

    def _commit(self, v: float):
        v = max(self.from_, min(self.to, round((v - self.from_) / self.step) * self.step + self.from_))
        v = round(v, 6)
        if v != self._value():
            self.var.set(v)
            if self.on_change:
                self.on_change()

    def _from_mouse(self, e):
        self.focus_set()
        x0, x1 = self._bounds()
        frac = max(0.0, min(1.0, (e.x - x0) / max(1, x1 - x0)))
        self._commit(self.from_ + frac * (self.to - self.from_))

    def _nudge(self, d: int):
        self._commit(self._value() + d * self.step)

    def draw(self):
        self.delete("all")
        h = self.winfo_height()
        x0, x1 = self._bounds()
        if x1 - x0 < 20:
            return
        v = self._value()
        frac = (v - self.from_) / max(1e-9, self.to - self.from_)
        fx = x0 + (x1 - x0) * max(0.0, min(1.0, frac))
        by0, by1 = h * 0.30, h * 0.72
        bh = by1 - by0
        skew = bh * 0.9
        step = max(5.0, bh * 0.72)
        sw = step * 0.55
        x = x0
        while x + sw + skew <= x1:
            col = self.pal.fg if (x + (sw + skew) / 2) <= fx else self.pal.dim
            self.create_polygon(x, by1, x + sw, by1, x + sw + skew, by0, x + skew, by0, fill=col, outline="")
            x += step
        self.create_line(fx, by0 - 4, fx, by1 + 4, fill=self.pal.fg, width=2)
        self.create_text(self.winfo_width() - 2, h / 2, anchor="e", text=self.fmt(v), font=self.f, fill=self.pal.fg)


class Segmented(tk.Frame):
    """Row of pill buttons bound to a StringVar."""

    def __init__(self, parent, pal: Palette, font: tkfont.Font, var: tk.StringVar,
                 options: Sequence[Tuple[str, str]], on_change: Callable[[], None] = None):
        super().__init__(parent, bg=pal.panel)
        self.pal, self.var, self.on_change = pal, var, on_change
        self._items: Dict[str, tk.Label] = {}
        for value, label in options:
            lb = tk.Label(self, text=label, font=font, padx=12, pady=5, cursor="hand2")
            lb.pack(side="left", padx=(0, 4))
            lb.bind("<Button-1>", lambda e, v=value: self._pick(v))
            lb.bind("<Enter>", lambda e, v=value: self._paint(hover=v))
            lb.bind("<Leave>", lambda e: self._paint())
            self._items[value] = lb
        var.trace_add("write", lambda *a: self._paint())
        self._paint()

    def _pick(self, v: str):
        if self.var.get() != v:
            self.var.set(v)
            if self.on_change:
                self.on_change()

    def _paint(self, hover: Optional[str] = None):
        cur = self.var.get()
        for v, lb in self._items.items():
            if v == cur:
                lb.configure(bg=self.pal.fg, fg=self.pal.bg)
            else:
                lb.configure(bg=self.pal.hover if v == hover else self.pal.field, fg=self.pal.fg)


class FlatButton(tk.Label):
    def __init__(self, parent, pal: Palette, font: tkfont.Font, text: str, command: Callable[[], None],
                 primary: bool = False, **kw):
        self.pal, self.primary = pal, primary
        super().__init__(parent, text=text, font=font, cursor="hand2", padx=kw.pop("padx", 18),
                         pady=kw.pop("pady", 7), **kw)
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self._paint(False)

    def _paint(self, hover: bool):
        if self.primary:
            self.configure(bg=self.pal.sub if hover else self.pal.fg, fg=self.pal.bg)
        else:
            self.configure(bg=self.pal.hover if hover else self.pal.field, fg=self.pal.fg)


class ColorField(tk.Frame):
    """Clickable swatch + hex entry."""

    def __init__(self, parent, pal: Palette, font: tkfont.Font, var: tk.StringVar, title: str,
                 on_change: Callable[[], None] = None):
        super().__init__(parent, bg=pal.panel)
        self.var, self.title, self.on_change = var, title, on_change
        self.swatch = tk.Frame(self, width=34, height=24, cursor="hand2", highlightthickness=1,
                               highlightbackground=pal.line)
        self.swatch.pack(side="left", padx=(0, 10))
        self.swatch.bind("<Button-1>", lambda e: self._pick())
        ent = tk.Entry(self, textvariable=var, font=font, width=9, relief="flat", bg=pal.field, fg=pal.fg,
                       insertbackground=pal.fg, highlightthickness=1, highlightbackground=pal.field,
                       highlightcolor=pal.line)
        ent.pack(side="left", ipady=3)
        var.trace_add("write", lambda *a: self._changed())
        self._paint()

    def _paint(self):
        c = self.var.get().strip()
        if _valid_hex(c):
            self.swatch.configure(bg=c)

    def _changed(self):
        self._paint()
        if _valid_hex(self.var.get().strip()) and self.on_change:
            self.on_change()

    def _pick(self):
        cur = self.var.get().strip()
        res = colorchooser.askcolor(color=cur if _valid_hex(cur) else None, title=self.title,
                                    parent=self.winfo_toplevel())
        if res and res[1]:
            self.var.set(res[1])


# ===================== settings window =====================
class SettingsWindow(tk.Toplevel):
    """Settings configuration window"""

    PAGES = ("Display", "Theme", "Layout", "Limits", "Sensors")

    def __init__(self, parent: "DashboardTk"):
        super().__init__(parent)
        self.parent = parent
        self.title("Settings — HW Monitor")
        self.geometry("920x640")
        self.minsize(760, 480)

        # Effective config (defaults merged) as starting point, raw file to preserve unknown keys
        self.original = json.loads(json.dumps(parent.cfg))
        self.base = self._load_file() or json.loads(json.dumps(parent.cfg))
        eff = self.original

        th = eff.get("theme", {})
        self.pal = pal = Palette(th.get("bg", "#0b0e13"), th.get("fg", "#e5eef9"),
                                 th.get("sub", "#9fb0c6"), th.get("hot", "#ff5a4f"))
        fam = getattr(parent, "display_font", "Segoe UI")
        self.f_title = tkfont.Font(self, family=fam, size=-30, weight="bold")
        self.f_card = tkfont.Font(self, family=fam, size=-19, weight="bold")
        self.f_nav = tkfont.Font(self, family=fam, size=-17, weight="bold")
        self.f_text = tkfont.Font(self, family=fam, size=-15)
        self.f_bold = tkfont.Font(self, family=fam, size=-15, weight="bold")
        self.f_small = tkfont.Font(self, family=fam, size=-13)
        self.configure(bg=pal.bg)

        try:
            icon_path = os.path.join(SCRIPT_DIR, "hwmonitor.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except Exception:
            pass

        self._style_ttk()

        self.v: Dict[str, tk.Variable] = {}
        self.sensor_combos: Dict[str, ttk.Combobox] = {}
        self.available_sensors: List[Dict] = []
        self._preview_job = None
        self._saved = False

        self._build_shell()
        self._pages: Dict[str, tk.Frame] = {}
        for name in self.PAGES:
            page = tk.Frame(self.scroll_inner, bg=pal.bg)
            getattr(self, f"_page_{name.lower()}")(page, eff)
            self._pages[name] = page
        self._show_page("Display")

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda e: self._cancel())
        self.bind_all("<MouseWheel>", self._on_wheel)

        # Modal, on top of the (possibly fullscreen) dashboard
        self.transient(parent)
        self.attributes("-topmost", True)
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        try:
            self.grab_set()
        except tk.TclError:
            self.after(100, lambda: self.winfo_exists() and self.grab_set())
        self.focus_force()

        self.after(50, self._load_sensors)

    # ---------- config file ----------
    @staticmethod
    def _load_file() -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ---------- shell ----------
    def _style_ttk(self):
        pal = self.pal
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("HW.TCombobox", fieldbackground=pal.field, background=pal.field, foreground=pal.fg,
                     arrowcolor=pal.fg, bordercolor=pal.field, lightcolor=pal.field, darkcolor=pal.field,
                     selectbackground=pal.field, selectforeground=pal.fg, padding=5)
        st.map("HW.TCombobox", fieldbackground=[("readonly", pal.field)], foreground=[("readonly", pal.fg)],
               background=[("active", pal.hover)], arrowcolor=[("active", pal.fg)])
        self.option_add("*TCombobox*Listbox.background", pal.field)
        self.option_add("*TCombobox*Listbox.foreground", pal.fg)
        self.option_add("*TCombobox*Listbox.selectBackground", pal.fg)
        self.option_add("*TCombobox*Listbox.selectForeground", pal.bg)
        self.option_add("*TCombobox*Listbox.font", self.f_text)

    def _build_shell(self):
        pal = self.pal
        # Sidebar
        side = tk.Frame(self, bg=pal.panel, width=190)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="SETTINGS", font=self.f_title, bg=pal.panel, fg=pal.fg).pack(anchor="w", padx=20, pady=(22, 2))
        cv = tk.Canvas(side, height=10, bg=pal.panel, highlightthickness=0)
        cv.pack(fill="x", padx=20, pady=(0, 18))
        cv.bind("<Configure>", lambda e: (cv.delete("all"), cv.create_line(0, 5, e.width - 6, 5, fill=pal.fg),
                                          cv.create_oval(e.width - 9, 2, e.width - 3, 8, fill=pal.fg, outline="")))
        self._nav: Dict[str, tk.Label] = {}
        for name in self.PAGES:
            lb = tk.Label(side, text=name, font=self.f_nav, bg=pal.panel, fg=pal.sub, anchor="w",
                          padx=20, pady=9, cursor="hand2")
            lb.pack(fill="x")
            lb.bind("<Button-1>", lambda e, n=name: self._show_page(n))
            lb.bind("<Enter>", lambda e, l=lb, n=name: n != self._current and l.configure(bg=pal.hover))
            lb.bind("<Leave>", lambda e, l=lb, n=name: n != self._current and l.configure(bg=pal.panel))
            self._nav[name] = lb
        tk.Label(side, text="Changes preview live\non the dashboard.", font=self.f_small, bg=pal.panel,
                 fg=pal.sub, justify="left").pack(side="bottom", anchor="w", padx=20, pady=20)

        # Footer
        foot = tk.Frame(self, bg=pal.bg)
        foot.pack(side="bottom", fill="x", padx=24, pady=(6, 16))
        FlatButton(foot, pal, self.f_bold, "Save", self._save, primary=True).pack(side="right")
        FlatButton(foot, pal, self.f_bold, "Cancel", self._cancel).pack(side="right", padx=(0, 8))
        self.status = tk.Label(foot, text="", font=self.f_small, bg=pal.bg, fg=pal.sub)
        self.status.pack(side="left")

        # Scrollable content
        body = tk.Frame(self, bg=pal.bg)
        body.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=(14, 0))
        self.scroll = tk.Canvas(body, bg=pal.bg, highlightthickness=0)
        self.scroll.pack(side="left", fill="both", expand=True)
        self.scroll_inner = tk.Frame(self.scroll, bg=pal.bg)
        self._win = self.scroll.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        self.scroll_inner.bind("<Configure>", lambda e: self.scroll.configure(scrollregion=self.scroll.bbox("all")))
        self.scroll.bind("<Configure>", lambda e: self.scroll.itemconfigure(self._win, width=e.width - 6))
        self._current = ""

    def _show_page(self, name: str):
        pal = self.pal
        for n, page in self._pages.items():
            page.pack_forget()
        self._pages[name].pack(fill="both", expand=True)
        self._current = name
        for n, lb in self._nav.items():
            sel = n == name
            lb.configure(bg=pal.bg if sel else pal.panel, fg=pal.fg if sel else pal.sub)
        self.scroll.yview_moveto(0)

    def _on_wheel(self, e):
        w = e.widget
        if isinstance(w, str) or not str(w).startswith(str(self)):
            return
        if self.scroll_inner.winfo_reqheight() > self.scroll.winfo_height():
            self.scroll.yview_scroll(int(-e.delta / 120), "units")

    def destroy(self):
        try:
            self.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass
        super().destroy()

    # ---------- building blocks ----------
    def _card(self, page, title: str, hint: str = "") -> tk.Frame:
        pal = self.pal
        card = tk.Frame(page, bg=pal.panel)
        card.pack(fill="x", pady=(0, 14), padx=(0, 8))
        head = tk.Frame(card, bg=pal.panel)
        head.pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(head, text=title, font=self.f_card, bg=pal.panel, fg=pal.fg).pack(side="left")
        if hint:
            tk.Label(head, text=hint, font=self.f_small, bg=pal.panel, fg=pal.sub).pack(side="left", padx=(12, 0), pady=(4, 0))
        cv = tk.Canvas(card, height=8, bg=pal.panel, highlightthickness=0)
        cv.pack(fill="x", padx=18, pady=(4, 6))
        cv.bind("<Configure>", lambda e: (cv.delete("all"), cv.create_oval(0, 1, 6, 7, fill=pal.fg, outline=""),
                                          cv.create_line(3, 4, e.width, 4, fill=pal.line)))
        inner = tk.Frame(card, bg=pal.panel)
        inner.pack(fill="x", padx=18, pady=(0, 14))
        inner.grid_columnconfigure(1, weight=1)
        return inner

    def _row(self, card: tk.Frame, label: str, widget_factory: Callable[[tk.Frame], tk.Widget], hint: str = ""):
        pal = self.pal
        r = card.grid_size()[1]
        lf = tk.Frame(card, bg=pal.panel)
        lf.grid(row=r, column=0, sticky="nw", pady=6, padx=(0, 16))
        tk.Label(lf, text=label, font=self.f_bold, bg=pal.panel, fg=pal.fg, anchor="w", width=17).pack(anchor="w")
        if hint:
            tk.Label(lf, text=hint, font=self.f_small, bg=pal.panel, fg=pal.sub, anchor="w",
                     justify="left", wraplength=170).pack(anchor="w")
        holder = tk.Frame(card, bg=pal.panel)
        holder.grid(row=r, column=1, sticky="ew", pady=6)
        w = widget_factory(holder)
        w.pack(fill="x", expand=True, anchor="w")
        return w

    def _slider(self, card, key: str, label: str, value: float, from_, to, step=1, fmt=None, hint=""):
        var = tk.DoubleVar(value=value)
        self.v[key] = var
        return self._row(card, label, lambda p: HatchSlider(p, self.pal, self.f_bold, var, from_, to, step, fmt,
                                                            self._changed), hint)

    def _segmented(self, card, key: str, label: str, value: str, options, hint=""):
        var = tk.StringVar(value=value)
        self.v[key] = var
        return self._row(card, label, lambda p: Segmented(p, self.pal, self.f_text, var, options, self._changed), hint)

    def _color(self, card, key: str, label: str, value: str, hint=""):
        var = tk.StringVar(value=value)
        self.v[key] = var
        return self._row(card, label, lambda p: ColorField(p, self.pal, self.f_text, var, label, self._changed), hint)

    def _entry(self, card, key: str, label: str, value: str, hint=""):
        pal = self.pal
        var = tk.StringVar(value=value)
        self.v[key] = var
        var.trace_add("write", lambda *a: self._changed())

        def make(p):
            return tk.Entry(p, textvariable=var, font=self.f_text, relief="flat", bg=pal.field, fg=pal.fg,
                            insertbackground=pal.fg, highlightthickness=1, highlightbackground=pal.field,
                            highlightcolor=pal.line)
        e = self._row(card, label, make, hint)
        e.pack_configure(ipady=5)
        return e

    def _combo(self, card, label: str, values: List[str], hint="") -> ttk.Combobox:
        return self._row(card, label, lambda p: ttk.Combobox(p, values=values, state="readonly",
                                                             style="HW.TCombobox", font=self.f_text), hint)

    # ---------- pages ----------
    def _page_display(self, page, eff):
        win = eff.get("window", {})
        mon = win.get("monitor", {})
        c = self._card(page, "Window", "applied on Save")
        self._segmented(c, "window.mode", "Mode", win.get("mode", "fullscreen"),
                        [("fullscreen", "Fullscreen"), ("windowed", "Windowed")])
        try:
            w, h = (int(x) for x in str(win.get("geometry", "1280x720")).split("+")[0].split("x"))
        except Exception:
            w, h = 1280, 720
        self._slider(c, "win_w", "Window width", w, 640, 3840, 10, lambda v: f"{v:.0f} px", "windowed mode")
        self._slider(c, "win_h", "Window height", h, 360, 2160, 10, lambda v: f"{v:.0f} px")

        c = self._card(page, "Monitor", "applied on Save")
        self._segmented(c, "window.monitor.select", "Position", mon.get("select", "primary"),
                        [("primary", "Primary"), ("left", "Left"), ("right", "Right"),
                         ("top", "Top"), ("bottom", "Bottom"), ("index", "Index")])
        mons = self._monitors()
        self._slider(c, "window.monitor.index", "Monitor index", mon.get("index", 1), 1, max(2, len(mons)), 1,
                     lambda v: f"#{v:.0f}", "used when Position = Index")
        if mons:
            txt = "\n".join(f"#{i + 1}   {m['w']} x {m['h']}   at ({m['x']}, {m['y']})" + ("   · primary" if m.get("primary") else "")
                            for i, m in enumerate(mons))
            self._row(c, "Detected", lambda p: tk.Label(p, text=txt, font=self.f_text, bg=self.pal.panel,
                                                       fg=self.pal.sub, justify="left", anchor="w"))

        c = self._card(page, "Data")
        self._slider(c, "poll.interval_sec", "Refresh interval", _get(eff, "poll.interval_sec", 1.0), 0.25, 5, 0.25,
                     lambda v: f"{v:g} s")
        self._slider(c, "poll.history_seconds", "History length", _get(eff, "poll.history_seconds", 300), 60, 1800, 30,
                     lambda v: f"{v / 60:g} min", "used by Max / Avg and graphs")

    def _page_theme(self, page, eff):
        th = eff.get("theme", {})
        pal = self.pal
        c = self._card(page, "Presets", "click to apply")

        def make_presets(p):
            box = tk.Frame(p, bg=pal.panel)
            for name, cols in PRESETS.items():
                lb = tk.Label(box, text=name, font=self.f_bold, bg=cols["bg"], fg=cols["fg"], padx=16, pady=8,
                              cursor="hand2", highlightthickness=2, highlightbackground=cols["sub"])
                lb.pack(side="left", padx=(0, 8))
                lb.bind("<Button-1>", lambda e, cs=cols: self._apply_preset(cs))
            return box
        self._row(c, "Color scheme", make_presets)

        c = self._card(page, "Colors")
        self._color(c, "theme.bg", "Background", th.get("bg", "#0b0e13"))
        self._color(c, "theme.fg", "Primary", th.get("fg", "#e5eef9"), "text, gauges, bars")
        self._color(c, "theme.sub", "Secondary", th.get("sub", "#9fb0c6"), "units, buttons")
        self._color(c, "theme.hot", "Alert", th.get("hot", "#ff5a4f"), "hot temperatures")

        c = self._card(page, "Typography")
        fams = sorted({f for f in tkfont.families(self) if not f.startswith("@")}, key=str.lower)
        cb = self._combo(c, "Font", fams)
        cur = th.get("display_font", "Bahnschrift")
        cb.set(cur if cur in fams else self.parent.display_font)
        cb.bind("<<ComboboxSelected>>", lambda e: self._changed())
        self.font_combo = cb
        self._slider(c, "fonts.clock_pt", "Clock size", _get(eff, "fonts.clock_pt", 40), 16, 72, 1, lambda v: f"{v:.0f}")
        self._slider(c, "theme.ui_scale", "UI scale", th.get("ui_scale", 1.0), 0.5, 2.0, 0.05,
                     lambda v: f"{v * 100:.0f}%", "scales everything")

    def _page_layout(self, page, eff):
        c = self._card(page, "Spacing")
        self._slider(c, "layout.outer_padding_px", "Outer margin", _get(eff, "layout.outer_padding_px", 10), 0, 60, 1,
                     lambda v: f"{v:.0f} px", "space around the dashboard")
        self._slider(c, "layout.inner_gap_px", "Panel gap", _get(eff, "layout.inner_gap_px", 5), 1, 30, 1,
                     lambda v: f"{v:.0f}", "space between panels")
        c = self._card(page, "Hotkeys", "applied on restart")
        hk = eff.get("hotkeys", {})
        self._entry(c, "hotkeys.fullscreen_toggle", "Fullscreen toggle", hk.get("fullscreen_toggle", "F11"))
        self._entry(c, "hotkeys.rescan_hwinfo", "Rescan sensors", hk.get("rescan_hwinfo", "R"))

    def _page_limits(self, page, eff):
        deg = lambda v: f"{v:.0f} °C"
        rpm = lambda v: f"{v:.0f} RPM"
        c = self._card(page, "Temperature scale", "value that fills the gauge")
        self._slider(c, "rings.cpu.vmax", "CPU", _get(eff, "rings.cpu.vmax", 100), 50, 150, 1, deg)
        self._slider(c, "rings.gpu.vmax", "GPU", _get(eff, "rings.gpu.vmax", 110), 50, 150, 1, deg)
        self._slider(c, "rings.mobo.vmax", "Mobo", _get(eff, "rings.mobo.vmax", 95), 30, 150, 1, deg)

        c = self._card(page, "Alert")
        self._slider(c, "theme.hot_ratio", "Hot threshold", _get(eff, "theme.hot_ratio", 0.85), 0.5, 1.0, 0.01,
                     lambda v: f"{v * 100:.0f}%", "of the scale above, turns the alert color on")

        c = self._card(page, "Graphs")
        self._slider(c, "graphs.mobo.max_value", "Mobo graph top", _get(eff, "graphs.mobo.max_value", 100), 30, 150, 1, deg,
                     "height of the MOBO history graph")

        c = self._card(page, "Fans", "value that fills the bar")
        self._slider(c, "fans.max_rpm", "Fans", _get(eff, "fans.max_rpm", 2000), 500, 5000, 100, rpm)
        self._slider(c, "fans.pump_max_rpm", "Pump", _get(eff, "fans.pump_max_rpm", 3500), 500, 6000, 100, rpm)

    def _page_sensors(self, page, eff):
        c = self._card(page, "Temperature sources", "applied on Save")
        for key, label in (("cpu", "CPU gauge"), ("gpu", "GPU gauge"), ("mobo", "Mobo panel")):
            cb = self._combo(c, label, [AUTO_SENSOR])
            cb.set(AUTO_SENSOR)
            self.sensor_combos[key] = cb
        self.sensor_status = tk.Label(c, text="Loading sensors…", font=self.f_small, bg=self.pal.panel, fg=self.pal.sub)
        self.sensor_status.grid(row=c.grid_size()[1], column=1, sticky="w", pady=(4, 0))
        self._row(c, "", lambda p: FlatButton(p, self.pal, self.f_text, "Reload list", self._load_sensors, padx=12, pady=4))

        c = self._card(page, "Names", "empty = automatic")
        ov = _get(eff, "labels.override", {}) or {}
        for key, label in (("cpu", "CPU name"), ("gpu", "GPU name"), ("mobo", "Mobo name")):
            self._entry(c, f"labels.override.{key}", label, ov.get(key) or "")

    # ---------- behaviour ----------
    @staticmethod
    def _monitors() -> List[dict]:
        try:
            mod = sys.modules.get("__main__")
            fn = getattr(mod, "list_monitors", None) or getattr(sys.modules.get("UI_Dashboard"), "list_monitors", None)
            return fn() if fn else []
        except Exception:
            return []

    def _apply_preset(self, cols: Dict[str, str]):
        for k, v in cols.items():
            self.v[f"theme.{k}"].set(v)
        self._changed()

    def _changed(self):
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(120, self._preview)

    def _preview(self):
        self._preview_job = None
        try:
            self.parent.apply_config(self._collect(for_preview=True))
            self.status.configure(text="Preview — not saved yet", fg=self.pal.sub)
        except Exception as e:
            self.status.configure(text=f"Preview error: {e}", fg=self.pal.hot)

    def _collect(self, for_preview: bool = False) -> dict:
        cfg = json.loads(json.dumps(self.base))
        ints = {"fonts.clock_pt", "layout.outer_padding_px", "layout.inner_gap_px", "window.monitor.index",
                "poll.history_seconds", "rings.cpu.vmax", "rings.gpu.vmax", "rings.mobo.vmax",
                "graphs.mobo.max_value", "fans.max_rpm", "fans.pump_max_rpm"}
        for key, var in self.v.items():
            if "." not in key:
                continue
            val = var.get()
            if key.startswith("theme.") and key.split(".")[1] in ("bg", "fg", "sub", "hot"):
                if not _valid_hex(str(val)):
                    continue
                val = str(val).strip()
            elif key.startswith("labels.override."):
                val = str(val).strip() or None
            elif key in ints:
                val = int(round(float(val)))
            elif isinstance(var, tk.DoubleVar):
                val = round(float(val), 4)
            _set(cfg, key, val)
        _set(cfg, "theme.display_font", self.font_combo.get() or self.parent.display_font)
        _set(cfg, "window.geometry", f"{int(self.v['win_w'].get())}x{int(self.v['win_h'].get())}+100+100")
        _set(cfg, "window.monitor.index_base", 1)

        # Sensor mapping (only on save — needs a rescan)
        mapping = cfg.get("sensor_mapping", {}) if isinstance(cfg.get("sensor_mapping"), dict) else {}
        if not for_preview and self.available_sensors:
            for key, cb in self.sensor_combos.items():
                idx = cb.current()
                if idx <= 0:
                    mapping.pop(key, None)  # Auto-detect
                elif idx - 1 < len(self.available_sensors):
                    s = self.available_sensors[idx - 1]
                    mapping[key] = {"sensor_index": s["sensor_index"], "entry_id": s["entry_id"], "label": s["label"]}
            cfg["sensor_mapping"] = mapping
        return cfg

    def _load_sensors(self):
        """Fill the sensor combos with every temperature entry exposed by HWiNFO (with live value)."""
        try:
            from hwreader import HwinfoReader
            reader = HwinfoReader()
            reader.open()
            try:
                reader.scan()
                sensors = reader.sensors or []
                entries = reader.entries or []
            finally:
                reader.close()
            temps = []
            for e in entries:
                if (e.get("unit") or "").strip().lower() not in ("°c", "c", "celsius"):
                    continue
                si = e.get("sensor_index", -1)
                sname = ""
                if 0 <= si < len(sensors):
                    sname = sensors[si].get("name_user") or sensors[si].get("name_original") or ""
                label = e.get("label_user") or e.get("label_original") or ""
                temps.append({"display": f"{label}  ({e.get('value', 0):.0f} °C)  —  {sname}",
                              "sensor_index": si, "entry_id": e.get("id"), "label": label})
            self.available_sensors = temps
            values = [AUTO_SENSOR] + [t["display"] for t in temps]
            mapping = self.original.get("sensor_mapping", {}) or {}
            for key, cb in self.sensor_combos.items():
                cb.configure(values=values)
                cur = mapping.get(key) or {}
                sel = 0
                for i, t in enumerate(temps):
                    if t["sensor_index"] == cur.get("sensor_index") and t["entry_id"] == cur.get("entry_id"):
                        sel = i + 1
                        break
                cb.current(sel)
            self.sensor_status.configure(text=f"{len(temps)} temperature sensors found", fg=self.pal.sub)
        except Exception as e:
            self.available_sensors = []
            self.sensor_status.configure(text=f"HWiNFO not available — {str(e)[:60]}", fg=self.pal.hot)

    def _save(self):
        cfg = self._collect()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.status.configure(text=f"Save failed: {e}", fg=self.pal.hot)
            return
        self._saved = True
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
            self._preview_job = None
        parent = self.parent
        self.destroy()
        parent.apply_config(cfg, place=True, rescan=True)

    def _cancel(self):
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
            self._preview_job = None
        parent = self.parent
        self.destroy()
        if not self._saved:
            parent.apply_config(self.original)


if __name__ == "__main__":
    # Standalone test: windowed dashboard + settings on top
    import UI_Dashboard as ui
    cfg = ui.load_config("hwconfig.json")
    app = ui.DashboardTk(cfg)
    app.geometry("1280x720+100+100")
    app.after(300, lambda: SettingsWindow(app))
    app.mainloop()
