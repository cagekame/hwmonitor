"""
Configuration: defaults, file location, load / save, dotted-path helpers.
Single source of truth for `hwconfig.json` — every other module goes through here.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Folder that holds hwconfig.json: next to the .exe when frozen (PyInstaller), else next to the sources.
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = APP_DIR

CONFIG_FILE = os.path.join(APP_DIR, "hwconfig.json")
ICON_FILE = os.path.join(RESOURCE_DIR, "hwmonitor.ico")

DEFAULT_CONFIG: dict = {
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
        "hot": "#ff5a4f",        # color used when a temperature is near its max
        "hot_ratio": 0.85,       # fraction of vmax above which "hot" kicks in
        "font_family": "Consolas",
        "display_font": "Bahnschrift",  # panel font (falls back to font_family if missing)
        "ui_scale": 1.0,
    },
    "fonts": {"clock_pt": 40},
    "poll": {"interval_sec": 1.0, "history_seconds": 300},
    "layout": {"outer_padding_px": 20, "inner_gap_px": 10},
    "rings": {"cpu": {"vmax": 100}, "mobo": {"vmax": 95}, "gpu": {"vmax": 110}},
    "graphs": {"mobo": {"max_value": 100}},
    "fans": {"max_rpm": 2000, "pump_max_rpm": 3500},
    "labels": {"override": {"cpu": None, "mobo": None, "gpu": None}},
    "hotkeys": {"fullscreen_toggle": "F11", "rescan_hwinfo": "R"},
    # HWiNFO free: shared memory stops after 12h → SYS panel turns red from `warn_hours`;
    # data not refreshed for `stale_after_sec` → OFFLINE, retry every `retry_sec`
    "hwinfo": {"warn_hours": 11.5, "stale_after_sec": 15, "retry_sec": 10,
               "process_names": ["HWiNFO64.exe", "HWiNFO32.exe", "HWiNFO.exe"]},
    # Temperature sources: {"sensor_index", "entry_id", "label"} — missing = auto-detect
    "sensor_mapping": {},
    # Fan sources: {"sensor_index", "entry_id", "label"} or just {"label"} — missing = auto-detect
    "fan_mapping": {},
}


def deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))


def deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def with_defaults(cfg: dict) -> dict:
    """New dict = DEFAULT_CONFIG overlaid with `cfg`."""
    return deep_merge(deep_copy(DEFAULT_CONFIG), deep_copy(cfg))


def load_raw(path: str = CONFIG_FILE) -> dict:
    """The file content as-is (no defaults); {} if missing or invalid."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def load_config(path: str = CONFIG_FILE) -> dict:
    """Effective configuration: defaults + file."""
    return with_defaults(load_raw(path))


def save_config(cfg: dict, path: str = CONFIG_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_path(d: dict, path: str, default: Any = None) -> Any:
    """get_path(cfg, "theme.bg")"""
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return default
        d = d[part]
    return d


def set_path(d: dict, path: str, value: Any) -> None:
    """set_path(cfg, "theme.bg", "#000000") — creates intermediate dicts."""
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(d.get(part), dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value
