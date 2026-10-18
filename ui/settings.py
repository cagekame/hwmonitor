"""
Settings window.
- Same visual language as the dashboard (theme colors, display font, hatched bars).
- Every change is previewed live on the dashboard; Save writes hwconfig.json, Cancel reverts.
- Window / monitor placement and sensor mapping are applied on Save.
Talks to the dashboard only through: app.cfg, app.display_font, app.apply_config().
"""

from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, colorchooser
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import config
import winsys
from theme import PRESETS, Theme, valid_hex

if TYPE_CHECKING:
    from ui.dashboard import DashboardApp

AUTO = "Auto-detect"


# ===================== custom widgets =====================
class HatchSlider(tk.Canvas):
    """Slider drawn like the dashboard bars. Click or drag to set; arrow keys nudge."""

    def __init__(self, parent, t: Theme, font: tkfont.Font, var: tk.DoubleVar,
                 from_: float, to: float, step: float = 1, fmt: Callable[[float], str] = None,
                 on_change: Callable[[], None] = None):
        super().__init__(parent, height=30, bg=t.panel, highlightthickness=0, cursor="hand2", takefocus=1)
        self.t, self.f, self.var = t, font, var
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
        return 4, max(10, self.winfo_width() - self._val_w - 12)

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
            col = self.t.fg if (x + (sw + skew) / 2) <= fx else self.t.dim
            self.create_polygon(x, by1, x + sw, by1, x + sw + skew, by0, x + skew, by0, fill=col, outline="")
            x += step
        self.create_line(fx, by0 - 4, fx, by1 + 4, fill=self.t.fg, width=2)
        self.create_text(self.winfo_width() - 2, h / 2, anchor="e", text=self.fmt(v), font=self.f, fill=self.t.fg)


class Segmented(tk.Frame):
    """Row of pill buttons bound to a StringVar."""

    def __init__(self, parent, t: Theme, font: tkfont.Font, var: tk.StringVar,
                 options: Sequence[Tuple[str, str]], on_change: Callable[[], None] = None):
        super().__init__(parent, bg=t.panel)
        self.t, self.var, self.on_change = t, var, on_change
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
                lb.configure(bg=self.t.fg, fg=self.t.bg)
            else:
                lb.configure(bg=self.t.hover if v == hover else self.t.input_bg, fg=self.t.fg)


class FlatButton(tk.Label):
    def __init__(self, parent, t: Theme, font: tkfont.Font, text: str, command: Callable[[], None],
                 primary: bool = False, **kw):
        self.t, self.primary = t, primary
        super().__init__(parent, text=text, font=font, cursor="hand2", padx=kw.pop("padx", 18),
                         pady=kw.pop("pady", 7), **kw)
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self._paint(False)

    def _paint(self, hover: bool):
        if self.primary:
            self.configure(bg=self.t.sub if hover else self.t.fg, fg=self.t.bg)
        else:
            self.configure(bg=self.t.hover if hover else self.t.input_bg, fg=self.t.fg)


class ColorField(tk.Frame):
    """Clickable swatch + hex entry."""

    def __init__(self, parent, t: Theme, font: tkfont.Font, var: tk.StringVar, title: str,
                 on_change: Callable[[], None] = None):
        super().__init__(parent, bg=t.panel)
        self.var, self.title, self.on_change = var, title, on_change
        self.swatch = tk.Frame(self, width=34, height=24, cursor="hand2", highlightthickness=1,
                               highlightbackground=t.line)
        self.swatch.pack(side="left", padx=(0, 10))
        self.swatch.bind("<Button-1>", lambda e: self._pick())
        tk.Entry(self, textvariable=var, font=font, width=9, relief="flat", bg=t.input_bg, fg=t.fg,
                 insertbackground=t.fg, highlightthickness=1, highlightbackground=t.input_bg,
                 highlightcolor=t.line).pack(side="left", ipady=3)
        var.trace_add("write", lambda *a: self._changed())
        self._paint()

    def _paint(self):
        c = self.var.get().strip()
        if valid_hex(c):
            self.swatch.configure(bg=c)

    def _changed(self):
        self._paint()
        if valid_hex(self.var.get().strip()) and self.on_change:
            self.on_change()

    def _pick(self):
        cur = self.var.get().strip()
        res = colorchooser.askcolor(color=cur if valid_hex(cur) else None, title=self.title,
                                    parent=self.winfo_toplevel())
        if res and res[1]:
            self.var.set(res[1])


# ===================== settings window =====================
class SettingsWindow(tk.Toplevel):
    PAGES = ("Display", "Theme", "Layout", "Limits", "Sensors")

    def __init__(self, app: "DashboardApp"):
        super().__init__(app)
        self.app = app
        self.title("Settings — HW Monitor")
        self.geometry("920x640")
        self.minsize(760, 480)

        # effective config (to show / revert) + raw file (to preserve keys we don't edit)
        self.original = config.deep_copy(app.cfg)
        self.base = config.load_raw() or config.deep_copy(app.cfg)
        eff = self.original

        self.t = t = Theme.from_config(eff)
        fam = app.display_font
        self.f_title = tkfont.Font(self, family=fam, size=-30, weight="bold")
        self.f_card = tkfont.Font(self, family=fam, size=-19, weight="bold")
        self.f_nav = tkfont.Font(self, family=fam, size=-17, weight="bold")
        self.f_text = tkfont.Font(self, family=fam, size=-15)
        self.f_bold = tkfont.Font(self, family=fam, size=-15, weight="bold")
        self.f_small = tkfont.Font(self, family=fam, size=-13)
        self.configure(bg=t.bg)
        try:
            if os.path.exists(config.ICON_FILE):
                self.iconbitmap(config.ICON_FILE)
        except Exception:
            pass

        self._style_ttk()

        self.v: Dict[str, tk.Variable] = {}
        # role → combobox, and the entries listed in it (index 0 = Auto-detect)
        self.temp_combos: Dict[str, ttk.Combobox] = {}
        self.fan_combos: Dict[str, ttk.Combobox] = {}
        self.temp_entries: List[dict] = []
        self.fan_entries: List[dict] = []
        self._sensors_loaded = False
        self._preview_job = None
        self._saved = False

        self._build_shell()
        self._pages: Dict[str, tk.Frame] = {}
        for name in self.PAGES:
            page = tk.Frame(self.scroll_inner, bg=t.bg)
            getattr(self, f"_page_{name.lower()}")(page, eff)
            self._pages[name] = page
        self._show_page("Display")

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda e: self._cancel())
        self.bind_all("<MouseWheel>", self._on_wheel)

        # Modal, on top of the (possibly fullscreen) dashboard
        self.transient(app)
        self.attributes("-topmost", True)
        self.update_idletasks()
        x = app.winfo_rootx() + (app.winfo_width() - self.winfo_width()) // 2
        y = app.winfo_rooty() + (app.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        try:
            self.grab_set()
        except tk.TclError:
            self.after(100, lambda: self.winfo_exists() and self.grab_set())
        self.focus_force()

        self.after(50, self._load_sensors)

    # ---------- shell ----------
    def _style_ttk(self):
        t = self.t
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("HW.TCombobox", fieldbackground=t.input_bg, background=t.input_bg, foreground=t.fg,
                     arrowcolor=t.fg, bordercolor=t.input_bg, lightcolor=t.input_bg, darkcolor=t.input_bg,
                     selectbackground=t.input_bg, selectforeground=t.fg, padding=5)
        st.map("HW.TCombobox", fieldbackground=[("readonly", t.input_bg)], foreground=[("readonly", t.fg)],
               background=[("active", t.hover)], arrowcolor=[("active", t.fg)])
        self.option_add("*TCombobox*Listbox.background", t.input_bg)
        self.option_add("*TCombobox*Listbox.foreground", t.fg)
        self.option_add("*TCombobox*Listbox.selectBackground", t.fg)
        self.option_add("*TCombobox*Listbox.selectForeground", t.bg)
        self.option_add("*TCombobox*Listbox.font", self.f_text)

    def _build_shell(self):
        t = self.t
        side = tk.Frame(self, bg=t.panel, width=190)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="SETTINGS", font=self.f_title, bg=t.panel, fg=t.fg).pack(anchor="w", padx=20, pady=(22, 2))
        cv = tk.Canvas(side, height=10, bg=t.panel, highlightthickness=0)
        cv.pack(fill="x", padx=20, pady=(0, 18))
        cv.bind("<Configure>", lambda e: (cv.delete("all"), cv.create_line(0, 5, e.width - 6, 5, fill=t.fg),
                                          cv.create_oval(e.width - 9, 2, e.width - 3, 8, fill=t.fg, outline="")))
        self._nav: Dict[str, tk.Label] = {}
        for name in self.PAGES:
            lb = tk.Label(side, text=name, font=self.f_nav, bg=t.panel, fg=t.sub, anchor="w",
                          padx=20, pady=9, cursor="hand2")
            lb.pack(fill="x")
            lb.bind("<Button-1>", lambda e, n=name: self._show_page(n))
            lb.bind("<Enter>", lambda e, l=lb, n=name: n != self._current and l.configure(bg=t.hover))
            lb.bind("<Leave>", lambda e, l=lb, n=name: n != self._current and l.configure(bg=t.panel))
            self._nav[name] = lb
        tk.Label(side, text="Changes preview live\non the dashboard.", font=self.f_small, bg=t.panel,
                 fg=t.sub, justify="left").pack(side="bottom", anchor="w", padx=20, pady=20)

        foot = tk.Frame(self, bg=t.bg)
        foot.pack(side="bottom", fill="x", padx=24, pady=(6, 16))
        FlatButton(foot, t, self.f_bold, "Save", self._save, primary=True).pack(side="right")
        FlatButton(foot, t, self.f_bold, "Cancel", self._cancel).pack(side="right", padx=(0, 8))
        self.status = tk.Label(foot, text="", font=self.f_small, bg=t.bg, fg=t.sub)
        self.status.pack(side="left")

        body = tk.Frame(self, bg=t.bg)
        body.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=(14, 0))
        self.scroll = tk.Canvas(body, bg=t.bg, highlightthickness=0)
        self.scroll.pack(side="left", fill="both", expand=True)
        self.scroll_inner = tk.Frame(self.scroll, bg=t.bg)
        self._win = self.scroll.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        self.scroll_inner.bind("<Configure>", lambda e: self.scroll.configure(scrollregion=self.scroll.bbox("all")))
        self.scroll.bind("<Configure>", lambda e: self.scroll.itemconfigure(self._win, width=e.width - 6))
        self._current = ""

    def _show_page(self, name: str):
        t = self.t
        for page in self._pages.values():
            page.pack_forget()
        self._pages[name].pack(fill="both", expand=True)
        self._current = name
        for n, lb in self._nav.items():
            sel = n == name
            lb.configure(bg=t.bg if sel else t.panel, fg=t.fg if sel else t.sub)
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
        t = self.t
        card = tk.Frame(page, bg=t.panel)
        card.pack(fill="x", pady=(0, 14), padx=(0, 8))
        head = tk.Frame(card, bg=t.panel)
        head.pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(head, text=title, font=self.f_card, bg=t.panel, fg=t.fg).pack(side="left")
        if hint:
            tk.Label(head, text=hint, font=self.f_small, bg=t.panel, fg=t.sub).pack(side="left", padx=(12, 0), pady=(4, 0))
        cv = tk.Canvas(card, height=8, bg=t.panel, highlightthickness=0)
        cv.pack(fill="x", padx=18, pady=(4, 6))
        cv.bind("<Configure>", lambda e: (cv.delete("all"), cv.create_oval(0, 1, 6, 7, fill=t.fg, outline=""),
                                          cv.create_line(3, 4, e.width, 4, fill=t.line)))
        inner = tk.Frame(card, bg=t.panel)
        inner.pack(fill="x", padx=18, pady=(0, 14))
        inner.grid_columnconfigure(1, weight=1)
        return inner

    def _row(self, card: tk.Frame, label: str, widget_factory: Callable[[tk.Frame], tk.Widget], hint: str = ""):
        t = self.t
        r = card.grid_size()[1]
        lf = tk.Frame(card, bg=t.panel)
        lf.grid(row=r, column=0, sticky="nw", pady=6, padx=(0, 16))
        tk.Label(lf, text=label, font=self.f_bold, bg=t.panel, fg=t.fg, anchor="w", width=17).pack(anchor="w")
        if hint:
            tk.Label(lf, text=hint, font=self.f_small, bg=t.panel, fg=t.sub, anchor="w",
                     justify="left", wraplength=170).pack(anchor="w")
        holder = tk.Frame(card, bg=t.panel)
        holder.grid(row=r, column=1, sticky="ew", pady=6)
        w = widget_factory(holder)
        w.pack(fill="x", expand=True, anchor="w")
        return w

    def _slider(self, card, key: str, label: str, value: float, from_, to, step=1, fmt=None, hint=""):
        var = tk.DoubleVar(value=value)
        self.v[key] = var
        return self._row(card, label, lambda p: HatchSlider(p, self.t, self.f_bold, var, from_, to, step, fmt,
                                                            self._changed), hint)

    def _segmented(self, card, key: str, label: str, value: str, options, hint=""):
        var = tk.StringVar(value=value)
        self.v[key] = var
        return self._row(card, label, lambda p: Segmented(p, self.t, self.f_text, var, options, self._changed), hint)

    def _color(self, card, key: str, label: str, value: str, hint=""):
        var = tk.StringVar(value=value)
        self.v[key] = var
        return self._row(card, label, lambda p: ColorField(p, self.t, self.f_text, var, label, self._changed), hint)

    def _entry(self, card, key: str, label: str, value: str, hint=""):
        t = self.t
        var = tk.StringVar(value=value)
        self.v[key] = var
        var.trace_add("write", lambda *a: self._changed())

        def make(p):
            return tk.Entry(p, textvariable=var, font=self.f_text, relief="flat", bg=t.input_bg, fg=t.fg,
                            insertbackground=t.fg, highlightthickness=1, highlightbackground=t.input_bg,
                            highlightcolor=t.line)
        e = self._row(card, label, make, hint)
        e.pack_configure(ipady=5)
        return e

    def _combo(self, card, label: str, values: List[str], hint="") -> ttk.Combobox:
        return self._row(card, label, lambda p: ttk.Combobox(p, values=values, state="readonly",
                                                             style="HW.TCombobox", font=self.f_text), hint)

    # ---------- pages ----------
    def _page_display(self, page, eff):
        g = lambda path, dv: config.get_path(eff, path, dv)
        c = self._card(page, "Window", "applied on Save")
        self._segmented(c, "window.mode", "Mode", g("window.mode", "fullscreen"),
                        [("fullscreen", "Fullscreen"), ("windowed", "Windowed")])
        try:
            w, h = (int(x) for x in str(g("window.geometry", "1280x720")).split("+")[0].split("x"))
        except Exception:
            w, h = 1280, 720
        self._slider(c, "win_w", "Window width", w, 640, 3840, 10, lambda v: f"{v:.0f} px", "windowed mode")
        self._slider(c, "win_h", "Window height", h, 360, 2160, 10, lambda v: f"{v:.0f} px")

        c = self._card(page, "Monitor", "applied on Save")
        self._segmented(c, "window.monitor.select", "Position", g("window.monitor.select", "primary"),
                        [("primary", "Primary"), ("left", "Left"), ("right", "Right"),
                         ("top", "Top"), ("bottom", "Bottom"), ("index", "Index")])
        mons = winsys.list_monitors()
        self._slider(c, "window.monitor.index", "Monitor index", g("window.monitor.index", 1), 1, max(2, len(mons)), 1,
                     lambda v: f"#{v:.0f}", "used when Position = Index")
        if mons:
            txt = "\n".join(f"#{i + 1}   {m['w']} x {m['h']}   at ({m['x']}, {m['y']})" + ("   · primary" if m.get("primary") else "")
                            for i, m in enumerate(mons))
            self._row(c, "Detected", lambda p: tk.Label(p, text=txt, font=self.f_text, bg=self.t.panel,
                                                       fg=self.t.sub, justify="left", anchor="w"))

        c = self._card(page, "Data")
        self._slider(c, "poll.interval_sec", "Refresh interval", g("poll.interval_sec", 1.0), 0.25, 5, 0.25,
                     lambda v: f"{v:g} s")
        self._slider(c, "poll.history_seconds", "History length", g("poll.history_seconds", 300), 60, 1800, 30,
                     lambda v: f"{v / 60:g} min", "used by Max / Avg and graphs")

    def _page_theme(self, page, eff):
        th = eff.get("theme", {})
        t = self.t
        c = self._card(page, "Presets", "click to apply")

        def make_presets(p):
            box = tk.Frame(p, bg=t.panel)
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
        cb.set(cur if cur in fams else self.app.display_font)
        cb.bind("<<ComboboxSelected>>", lambda e: self._changed())
        self.font_combo = cb
        self._slider(c, "fonts.clock_pt", "Clock size", config.get_path(eff, "fonts.clock_pt", 40), 16, 72, 1,
                     lambda v: f"{v:.0f}")
        self._slider(c, "theme.ui_scale", "UI scale", th.get("ui_scale", 1.0), 0.5, 2.0, 0.05,
                     lambda v: f"{v * 100:.0f}%", "scales everything")

    def _page_layout(self, page, eff):
        g = lambda path, dv: config.get_path(eff, path, dv)
        c = self._card(page, "Spacing")
        self._slider(c, "layout.outer_padding_px", "Outer margin", g("layout.outer_padding_px", 10), 0, 60, 1,
                     lambda v: f"{v:.0f} px", "space around the dashboard")
        self._slider(c, "layout.inner_gap_px", "Panel gap", g("layout.inner_gap_px", 5), 1, 30, 1,
                     lambda v: f"{v:.0f}", "space between panels")
        c = self._card(page, "Hotkeys", "applied on restart")
        self._entry(c, "hotkeys.fullscreen_toggle", "Fullscreen toggle", g("hotkeys.fullscreen_toggle", "F11"))
        self._entry(c, "hotkeys.rescan_hwinfo", "Rescan sensors", g("hotkeys.rescan_hwinfo", "R"))

    def _page_limits(self, page, eff):
        g = lambda path, dv: config.get_path(eff, path, dv)
        deg = lambda v: f"{v:.0f} °C"
        rpm = lambda v: f"{v:.0f} RPM"
        c = self._card(page, "Temperature scale", "value that fills the gauge")
        self._slider(c, "rings.cpu.vmax", "CPU", g("rings.cpu.vmax", 100), 50, 150, 1, deg)
        self._slider(c, "rings.gpu.vmax", "GPU", g("rings.gpu.vmax", 110), 50, 150, 1, deg)
        self._slider(c, "rings.mobo.vmax", "Mobo", g("rings.mobo.vmax", 95), 30, 150, 1, deg)

        c = self._card(page, "Alerts")
        self._slider(c, "theme.hot_ratio", "Hot threshold", g("theme.hot_ratio", 0.85), 0.5, 1.0, 0.01,
                     lambda v: f"{v * 100:.0f}%", "of the temperature scale")
        self._slider(c, "hwinfo.warn_hours", "HWiNFO restart", g("hwinfo.warn_hours", 11.5), 1, 12, 0.25,
                     lambda v: f"{int(v)}:{int(round(v % 1 * 60)):02d} h",
                     "SYS turns red after this time (free HWiNFO stops at 12h)")

        c = self._card(page, "Graphs")
        self._slider(c, "graphs.mobo.max_value", "Mobo graph top", g("graphs.mobo.max_value", 100), 30, 150, 1, deg,
                     "height of the MOBO history graph")

        c = self._card(page, "Fans", "value that fills the bar")
        self._slider(c, "fans.max_rpm", "Fans", g("fans.max_rpm", 2000), 500, 5000, 100, rpm)
        self._slider(c, "fans.pump_max_rpm", "Pump", g("fans.pump_max_rpm", 3500), 500, 6000, 100, rpm)

    def _page_sensors(self, page, eff):
        c = self._card(page, "Temperature sources", "applied on Save")
        for role, label in (("cpu", "CPU gauge"), ("gpu", "GPU gauge"), ("mobo", "Mobo panel")):
            self.temp_combos[role] = cb = self._combo(c, label, [AUTO])
            cb.set(AUTO)

        c = self._card(page, "Fan sources", "applied on Save")
        for role, label in (("cpu", "CPU fan"), ("gpu", "GPU fan"), ("pump", "Pump")):
            self.fan_combos[role] = cb = self._combo(c, label, [AUTO])
            cb.set(AUTO)
        self.sensor_status = tk.Label(c, text="Loading sensors…", font=self.f_small, bg=self.t.panel, fg=self.t.sub)
        self.sensor_status.grid(row=c.grid_size()[1], column=1, sticky="w", pady=(4, 0))
        self._row(c, "", lambda p: FlatButton(p, self.t, self.f_text, "Reload list", self._load_sensors, padx=12, pady=4))

        c = self._card(page, "Names", "empty = automatic")
        ov = config.get_path(eff, "labels.override", {}) or {}
        for key, label in (("cpu", "CPU name"), ("gpu", "GPU name"), ("mobo", "Mobo name")):
            self._entry(c, f"labels.override.{key}", label, ov.get(key) or "")

    # ---------- behaviour ----------
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
            self.app.apply_config(self._collect(for_preview=True))
            self.status.configure(text="Preview — not saved yet", fg=self.t.sub)
        except Exception as e:
            self.status.configure(text=f"Preview error: {e}", fg=self.t.hot)

    def _collect(self, for_preview: bool = False) -> dict:
        cfg = config.deep_copy(self.base)
        ints = {"fonts.clock_pt", "layout.outer_padding_px", "layout.inner_gap_px", "window.monitor.index",
                "poll.history_seconds", "rings.cpu.vmax", "rings.gpu.vmax", "rings.mobo.vmax",
                "graphs.mobo.max_value", "fans.max_rpm", "fans.pump_max_rpm"}
        for key, var in self.v.items():
            if "." not in key:
                continue
            val = var.get()
            if key.startswith("theme.") and key.split(".")[1] in ("bg", "fg", "sub", "hot"):
                if not valid_hex(str(val)):
                    continue
                val = str(val).strip()
            elif key.startswith("labels.override."):
                val = str(val).strip() or None
            elif key in ints:
                val = int(round(float(val)))
            elif isinstance(var, tk.DoubleVar):
                val = round(float(val), 4)
            config.set_path(cfg, key, val)
        config.set_path(cfg, "theme.display_font", self.font_combo.get() or self.app.display_font)
        config.set_path(cfg, "window.geometry", f"{int(self.v['win_w'].get())}x{int(self.v['win_h'].get())}+100+100")
        config.set_path(cfg, "window.monitor.index_base", 1)

        # Sensor / fan mapping: only on save (needs a rescan) and only if the lists were loaded
        if not for_preview and self._sensors_loaded:
            for section, combos, entries in (("sensor_mapping", self.temp_combos, self.temp_entries),
                                             ("fan_mapping", self.fan_combos, self.fan_entries)):
                mapping = cfg.get(section) if isinstance(cfg.get(section), dict) else {}
                for role, cb in combos.items():
                    idx = cb.current()
                    if idx <= 0:
                        mapping.pop(role, None)  # Auto-detect
                    elif idx - 1 < len(entries):
                        e = entries[idx - 1]
                        mapping[role] = {"sensor_index": e["sensor_index"], "entry_id": e["entry_id"], "label": e["label"]}
                cfg[section] = mapping
        return cfg

    def _load_sensors(self):
        """Fill the source combos with HWiNFO entries (°C for temperatures, RPM for fans) and their live value."""
        try:
            from hwreader import HwinfoReader
            reader = HwinfoReader()
            reader.open()
            try:
                reader.scan()
                self.temp_entries = reader.list_entries(["°C", "C", "Celsius"])
                self.fan_entries = reader.list_entries(["RPM", "R.P.M."])
            finally:
                reader.close()
        except Exception as e:
            self._sensors_loaded = False
            self.sensor_status.configure(text=f"HWiNFO not available — {str(e)[:60]}", fg=self.t.hot)
            return

        def fill(combos, entries, mapping, unit):
            values = [AUTO] + [f"{e['label']}  ({(e['value'] or 0):.0f} {unit})  —  {e['sensor']}" for e in entries]
            for role, cb in combos.items():
                cb.configure(values=values)
                cur = mapping.get(role) or {}
                sel = 0
                for i, e in enumerate(entries):
                    if cur.get("sensor_index") is not None:
                        hit = e["sensor_index"] == cur.get("sensor_index") and e["entry_id"] == cur.get("entry_id")
                    else:
                        hit = bool(cur.get("label")) and e["label"].lower() == str(cur["label"]).lower()
                    if hit:
                        sel = i + 1
                        break
                cb.current(sel)

        fill(self.temp_combos, self.temp_entries, self.original.get("sensor_mapping") or {}, "°C")
        fill(self.fan_combos, self.fan_entries, self.original.get("fan_mapping") or {}, "RPM")
        self._sensors_loaded = True
        self.sensor_status.configure(
            text=f"{len(self.temp_entries)} temperature and {len(self.fan_entries)} fan sensors found", fg=self.t.sub)

    def _save(self):
        cfg = self._collect()
        try:
            config.save_config(cfg)
        except Exception as e:
            self.status.configure(text=f"Save failed: {e}", fg=self.t.hot)
            return
        self._saved = True
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
            self._preview_job = None
        app = self.app
        self.destroy()
        app.apply_config(cfg, place=True, rescan=True)

    def _cancel(self):
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
            self._preview_job = None
        app = self.app
        self.destroy()
        if not self._saved:
            app.apply_config(self.original)
