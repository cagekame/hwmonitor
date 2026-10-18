"""
Theme: colors derived from the config, color math, presets.
Shared by the dashboard and the settings window (no module-level mutable globals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

OFFLINE = "#ff7a1a"

PRESETS: Dict[str, Dict[str, str]] = {
    "Dark":  {"bg": "#0b0e13", "fg": "#e5eef9", "sub": "#9fb0c6", "hot": "#ff5a4f"},
    "Light": {"bg": "#f3f4f6", "fg": "#111318", "sub": "#5b6472", "hot": "#d93025"},
    "Neon":  {"bg": "#06070c", "fg": "#3ee8ff", "sub": "#8b7dff", "hot": "#ff3fd2"},
    "Amber": {"bg": "#0d0a05", "fg": "#ffb547", "sub": "#a88a5c", "hot": "#ff4b3a"},
}


def valid_hex(c: str) -> bool:
    c = (c or "").strip()
    if len(c) != 7 or not c.startswith("#"):
        return False
    try:
        int(c[1:], 16)
        return True
    except ValueError:
        return False


def _rgb(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def blend(c1: str, c2: str, t: float) -> str:
    """Mix c1 → c2 by t (0..1). Falls back to c2 on non-hex colors."""
    try:
        a = _rgb(c1); b = _rgb(c2)
    except Exception:
        return c2
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


@dataclass
class Theme:
    bg: str = "#0b0e13"
    fg: str = "#e5eef9"
    sub: str = "#9fb0c6"
    hot: str = "#ff5a4f"
    offline: str = OFFLINE
    # derived tones (computed in __post_init__)
    dim: str = field(init=False)    # empty segments / stripes
    panel: str = field(init=False)  # settings: card background
    input_bg: str = field(init=False)  # settings: input background
    hover: str = field(init=False)  # settings: hover background
    line: str = field(init=False)   # settings: separators

    def __post_init__(self):
        self.dim = blend(self.bg, self.fg, 0.22)
        self.panel = blend(self.bg, self.fg, 0.04)
        self.input_bg = blend(self.bg, self.fg, 0.09)
        self.hover = blend(self.bg, self.fg, 0.14)
        self.line = blend(self.bg, self.fg, 0.30)

    @classmethod
    def from_config(cls, cfg: dict) -> "Theme":
        th = cfg.get("theme", {}) or {}
        d = cls()
        pick = lambda k, dv: th.get(k) if valid_hex(th.get(k, "")) else dv
        return cls(bg=pick("bg", d.bg), fg=pick("fg", d.fg), sub=pick("sub", d.sub), hot=pick("hot", d.hot))


def resolve_font(root, wanted: str, fallback: str = "Consolas") -> str:
    """`wanted` if installed, else `fallback`."""
    import tkinter.font as tkfont
    return wanted if wanted in tkfont.families(root) else fallback
