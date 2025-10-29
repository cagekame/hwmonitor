# dashboard_spacedesk_3rings_hwinfo_labels_fixed_pch.py
# Ring bianchi + sparkline + barre usage (CPU / RAM / GPU)
# RAM: usa "Memoria fisica usata" (+ "Memoria fisica totale" oppure "Memoria fisica disponibile")

import ctypes as ct
from ctypes import wintypes
import tkinter as tk
from datetime import datetime
import time
import re
import winreg
from collections import deque
import math
import platform

POLL_INTERVAL = 1.0
HISTORY_SECONDS = 300  # 5 minuti
BG = "#0b0e13"
FG = "#e5eef9"
SUB = "#9fb0c6"
GRID = "#16202f"
TRACK = "#151922"
FONT_FAMILY = "Consolas"

# ===================== DPI awareness =====================
try:
    user32 = ct.windll.user32
    shcore = ct.windll.shcore
    try:
        SetProcessDpiAwarenessContext = user32.SetProcessDpiAwarenessContext
        SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
        SetProcessDpiAwarenessContext(ct.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except Exception:
        try:
            shcore.SetProcessDpiAwareness(2)
        except Exception:
            user32.SetProcessDPIAware()
except Exception:
    pass

# ===================== HWiNFO SHM =====================
FILE_MAP_READ = 0x0004
MAP_NAMES = [
    "Global\\HWiNFO_SENS_SM2", "Global\\HWiNFO64_SENS_SM2",
    "HWiNFO_SENS_SM2", "HWiNFO64_SENS_SM2",
]
MAGIC_LITTLE = int.from_bytes(b"SiWH", "little")
MAGIC_BIG    = int.from_bytes(b"SiWH", "big")

class Hdr(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ct.c_uint32), ("version", ct.c_uint32), ("version2", ct.c_uint32),
        ("last_update", ct.c_int64), ("sensor_section_offset", ct.c_uint32),
        ("sensor_element_size", ct.c_uint32), ("sensor_element_count", ct.c_uint32),
        ("entry_section_offset", ct.c_uint32), ("entry_element_size", ct.c_uint32),
        ("entry_element_count", ct.c_uint32)
    ]

class Sensor(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ct.c_uint32), ("instance", ct.c_uint32),
        ("name_original", ct.c_char * 128), ("name_user", ct.c_char * 128)
    ]

class Entry(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", ct.c_uint32), ("sensor_index", ct.c_uint32), ("id", ct.c_uint32),
        ("name_original", ct.c_char * 128), ("name_user", ct.c_char * 128),
        ("unit", ct.c_char * 16), ("value", ct.c_double),
        ("value_min", ct.c_double), ("value_max", ct.c_double), ("value_avg", ct.c_double)
    ]

def k32():
    k = ct.windll.kernel32
    k.OpenFileMappingW.restype = wintypes.HANDLE
    k.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.OpenFileMappingA.restype = wintypes.HANDLE
    k.OpenFileMappingA.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCSTR]
    k.MapViewOfFile.restype = wintypes.LPVOID
    k.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ct.c_size_t]
    k.UnmapViewOfFile.restype = wintypes.BOOL
    k.UnmapViewOfFile.argtypes = [wintypes.LPCVOID]
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    return k

def open_mapping():
    if platform.system() != "Windows":
        raise RuntimeError("Richiede Windows")
    k = k32()
    for name in MAP_NAMES:
        h = k.OpenFileMappingW(FILE_MAP_READ, False, name)
        if not h:
            h = k.OpenFileMappingA(FILE_MAP_READ, False, name.encode("ascii", "ignore"))
        if not h:
            continue
        pv = k.MapViewOfFile(h, FILE_MAP_READ, 0, 0, 0)
        if pv:
            return k, h, ct.c_void_p(pv)
        k.CloseHandle(h)
    raise RuntimeError("HWiNFO Shared Memory non trovata (abilita ‘Shared Memory Support’ e apri la finestra Sensors).")

def close_mapping(k, h, view):
    try:
        if view: k.UnmapViewOfFile(view)
    except Exception: pass
    try:
        if h: k.CloseHandle(h)
    except Exception: pass

def cstr(b: bytes) -> str:
    i = b.find(b"\x00")
    if i != -1: b = b[:i]
    try:
        return b.decode("utf-8", "ignore")
    except:
        return ""

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def load_tables(view):
    base = int(view.value)
    hdr = Hdr.from_address(base)
    if hdr.magic not in (MAGIC_LITTLE, MAGIC_BIG):
        raise RuntimeError(f"Header non riconosciuto (magic=0x{hdr.magic:08X})")
    pS = base + hdr.sensor_section_offset
    sensors = []
    for i in range(hdr.sensor_element_count):
        s = Sensor.from_address(pS + i * hdr.sensor_element_size)
        sensors.append({
            "id": s.id, "instance": s.instance,
            "name_user": cstr(bytes(s.name_user)).strip(),
            "name_original": cstr(bytes(s.name_original)).strip(),
        })
    pE = base + hdr.entry_section_offset
    entries = []
    for i in range(hdr.entry_element_count):
        e = Entry.from_address(pE + i * hdr.entry_element_size)
        entries.append({
            "type": e.type, "sensor_index": e.sensor_index, "id": e.id,
            "label_user": cstr(bytes(e.name_user)).strip(),
            "label_original": cstr(bytes(e.name_original)).strip(),
            "unit": cstr(bytes(e.unit)).strip(),
            "value": float(e.value),
        })
    return sensors, entries

def _match_any(needles, hay):
    h = (hay or "").lower()
    return any(n and n.lower() in h for n in needles)

def _sensor_name(sensors, idx):
    if 0 <= idx < len(sensors):
        s = sensors[idx]
        return s["name_user"] or s["name_original"]
    return ""

def find_temp_entry(sensors, entries, *, label_needles, sensor_needles):
    for e in entries:
        lbl = (e["label_user"] or e["label_original"])
        sname = _sensor_name(sensors, e["sensor_index"])
        if _match_any(label_needles, lbl) and _match_any(sensor_needles, sname):
            return ((e["sensor_index"], e["id"]), sname, lbl)
    return (None, "", "")

def find_entry_by_exact_label_dual(sensors, entries, targets_norm):
    for e in entries:
        if _norm(e["label_user"]) in targets_norm or _norm(e["label_original"]) in targets_norm:
            sname = _sensor_name(sensors, e["sensor_index"])
            return ((e["sensor_index"], e["id"]), sname, e["label_user"] or e["label_original"])
    return (None, "", "")

def find_entry_fallback_with_second_if_dup(sensors, entries, *, label_needles, sensor_needles):
    cand = []
    for idx, e in enumerate(entries):
        lbl = (e["label_user"] or e["label_original"])
        sname = _sensor_name(sensors, e["sensor_index"])
        if _match_any(label_needles, lbl) and _match_any(sensor_needles, sname):
            cand.append((idx, e))
    if not cand:
        return (None, "", "")
    _, e = (cand[1] if len(cand) > 1 else cand[0])
    return ((e["sensor_index"], e["id"]), _sensor_name(sensors, e["sensor_index"]), e["label_user"] or e["label_original"])

def read_by_key(view, key):
    if not key: return None, None
    base = int(view.value)
    hdr = Hdr.from_address(base)
    pE = base + hdr.entry_section_offset
    for i in range(hdr.entry_element_count):
        e = Entry.from_address(pE + i * hdr.entry_element_size)
        if e.sensor_index == key[0] and e.id == key[1]:
            return float(e.value), (cstr(bytes(e.name_user)) or cstr(bytes(e.name_original)))
    return None, None

# ===================== label cleaners =====================
def _cleanup_name(s: str) -> str:
    s = re.sub(r"\(R\)|\(TM\)|\(tm\)|\(r\)|\(C\)", "", s or "")
    s = re.sub(r"\s{2,}", " ", s).strip(" -_/")
    return s.strip()

def clean_mobo_label(raw: str) -> str:
    s = raw or ""
    s = re.sub(r"^\s*Sistema:\s*", "", s, flags=re.I)
    s = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s)  # rimuove ( ... ) e [ ... ]
    s = re.sub(r"\s{2,}", " ", s).strip(" -_/")
    return s or "Motherboard / PCH"

# ===================== Registry names =====================
def get_cpu_name_reg() -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
            return _cleanup_name(val)
    except OSError:
        return None

def get_gpu_name_reg_discrete() -> str | None:
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    dgpu = []; igpu = []; others = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            i = 0
            while True:
                try: subname = winreg.EnumKey(root, i); i += 1
                except OSError: break
                if not re.fullmatch(r"\d{4}", subname): continue
                try:
                    with winreg.OpenKey(root, subname) as k:
                        desc, _ = winreg.QueryValueEx(k, "DriverDesc")
                        name = _cleanup_name(desc)
                        prov = ""
                        try: prov, _ = winreg.QueryValueEx(k, "ProviderName")
                        except OSError: pass
                        src = (prov + " " + name).lower()
                        if re.search(r"nvidia|geforce|amd|radeon", src): dgpu.append(name)
                        elif re.search(r"intel", src): igpu.append(name)
                        else: others.append(name)
                except OSError: continue
    except OSError:
        return None
    return (dgpu[0] if dgpu else (igpu[0] if igpu else (others[0] if others else None)))

# ===================== UI widgets =====================
class RingGauge(tk.Canvas):
    def __init__(self, parent, vmax=100.0, value_scale=0.28, label_scale=0.07, label_offset=1.08):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.maxv = float(vmax)
        self.value_scale = value_scale
        self.label_scale = label_scale      # ↑ aumentato (prima ~0.05)
        self.label_offset = label_offset    # ↑ avvicinato un po' al ring
        self.arc_id = self.val_id = self.bottom_label = None
        self.bind("<Configure>", self._resize)

    def _resize(self, _e=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        size = min(w, h) * 0.8
        cx, cy = w // 2, h // 2
        r = size / 2
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#151922", width=max(6, int(size * 0.04)))
        self.arc_id = self.create_arc(cx - r, cy - r, cx + r, cy + r, start=90, extent=0,
                                      style="arc", outline="#ffffff", width=max(10, int(size * 0.06)))
        self.val_id = self.create_text(cx, cy, text="--°", fill=FG,
                                       font=(FONT_FAMILY, int(size * self.value_scale), "bold"))
        self.bottom_label = self.create_text(cx, cy + r * self.label_offset, text="", fill=SUB,
                                             font=(FONT_FAMILY, int(size * self.label_scale)), anchor="n")

    def set(self, value, label_text):
        try:
            v = float(value)
            frac = max(0.0, min(1.0, v / max(1e-9, self.maxv)))
            self.itemconfigure(self.arc_id, extent=360 * frac)
            self.itemconfigure(self.val_id, text=f"{v:.0f}°")
        except Exception:
            self.itemconfigure(self.arc_id, extent=0)
            self.itemconfigure(self.val_id, text="--°")
        self.itemconfigure(self.bottom_label, text=label_text or "")

class TrendGraph(tk.Canvas):
    PAD_L = 24; PAD_R = 16; PAD_T = 18; PAD_B = 18
    def __init__(self, parent, max_seconds=HISTORY_SECONDS, max_value=120.0):
        super().__init__(parent, bg=BG, highlightthickness=0, height=120)
        self.max_seconds = max_seconds
        self.max_value = float(max_value)
        self.points = deque()
        self.bind("<Configure>", self._redraw)

    def push(self, value):
        t = time.time()
        try: v = float(value) if value is not None else None
        except Exception: v = None
        self.points.append((t, v))
        cutoff = t - self.max_seconds
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()
        self._redraw()

    def _redraw(self, _e=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 4 or h <= 4: return
        x0, y0, x1, y1 = self.PAD_L, self.PAD_T, w - self.PAD_R, h - self.PAD_B
        iw, ih = max(1, x1 - x0), max(1, y1 - y0)
        self.create_rectangle(x0, y0, x1, y1, outline=GRID, width=1)
        for xfrac in (0.25, 0.5, 0.75):
            x = int(x0 + iw * xfrac); self.create_line(x, y0, x, y1, fill=GRID)
        self.create_line(x0, int(y0 + ih * 0.5), x1, int(y0 + ih * 0.5), fill=GRID)
        if len(self.points) < 2: return
        t_now = self.points[-1][0]; t_min = t_now - self.max_seconds
        path = []
        for (t, v) in self.points:
            if t < t_min: continue
            x = int(x0 + (t - t_min) / self.max_seconds * (iw - 1))
            if v is None or math.isnan(v):
                if len(path) > 1: self.create_line(path, fill="#ffffff", width=2, smooth=True)
                path = []; continue
            y = int(y1 - (max(0.0, min(self.max_value, v)) / self.max_value) * (ih - 1))
            path.append((x, y))
        if len(path) > 1: self.create_line(path, fill="#ffffff", width=2, smooth=True)

class LoadBar(tk.Canvas):
    PAD_H = 16; PAD_V = 10
    def __init__(self, parent, height=50, label_text="Usage"):
        super().__init__(parent, bg=BG, highlightthickness=0, height=height)
        self.value = None; self.label_text = label_text
        self.bind("<Configure>", self._redraw)
    def set(self, value): self.value = value; self._redraw()
    def _redraw(self, _e=None):
        self.delete("all"); w = self.winfo_width(); h = self.winfo_height()
        if w < 10 or h < 10: return
        txt = f"{self.label_text}: --%"; frac = 0.0
        if self.value is not None:
            try:
                v = float(self.value); v = max(0.0, min(100.0, v)); frac = v / 100.0
                txt = f"{self.label_text}: {v:.0f}%"
            except Exception: pass
        self.create_text(self.PAD_H, self.PAD_V, anchor="w", text=txt, fill=FG, font=(FONT_FAMILY, 18, "bold"))
        x0, x1 = self.PAD_H, w - self.PAD_H
        y0, y1 = h - (self.PAD_V + 14), h - self.PAD_V
        self.create_rectangle(x0, y0, x1, y1, outline=TRACK, width=2)
        xf = x0 + int((x1 - x0) * frac)
        self.create_rectangle(x0+2, y0+2, max(x0+2, xf-2), y1-2, outline="", fill="#ffffff")

# ===================== Monitor =====================
class RECT(ct.Structure):
    _fields_ = [("left", ct.c_long), ("top", ct.c_long), ("right", ct.c_long), ("bottom", ct.c_long)]
class MONITORINFOEXW(ct.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT), ("rcWork", RECT),
                ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]
PMONITORENUMPROC = ct.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ct.POINTER(RECT), wintypes.LPARAM)
u32 = ct.windll.user32
u32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ct.POINTER(RECT), PMONITORENUMPROC, wintypes.LPARAM]
u32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ct.POINTER(MONITORINFOEXW)]
def list_monitors():
    mons = []
    def _cb(hMon, hdc, lprc, lparam):
        info = MONITORINFOEXW(); info.cbSize = ct.sizeof(MONITORINFOEXW)
        if u32.GetMonitorInfoW(hMon, ct.byref(info)):
            x, y = info.rcMonitor.left, info.rcMonitor.top
            w = info.rcMonitor.right - info.rcMonitor.left
            h = info.rcMonitor.bottom - info.rcMonitor.top
            mons.append({"device": info.szDevice, "x": x, "y": y, "w": w, "h": h})
        return True
    u32.EnumDisplayMonitors(0, None, PMONITORENUMPROC(_cb), 0)
    return mons
def pick_monitor(mons, name="spacedesk"):
    for m in mons:
        if name.lower() in m["device"].lower():
            return m
    return max(mons, key=lambda mm: (mm["x"], mm["y"]))

# ===================== App =====================
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg=BG); self.title("Dashboard — CPU / PCH / GPU")

        mons = list_monitors(); target = pick_monitor(mons)
        self.overrideredirect(True)
        self.geometry(f"{target['w']}x{target['h']}+{target['x']}+{target['y']}")
        self.update_idletasks()

        # Hotkeys
        self.bind("<Escape>", lambda e: self._quit())
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<r>", lambda e: self._reload_sensors()); self.bind("<R>", lambda e: self._reload_sensors())

        # Header clock
        header = tk.Frame(self, bg=BG, height=80); header.pack(side="top", fill="x")
        self.clock_lbl = tk.Label(header, text="", bg=BG, fg=FG, font=(FONT_FAMILY, 40, "bold"))
        self.clock_lbl.pack(anchor="center", pady=10)

        # Layout 3 righe
        container = tk.Frame(self, bg=BG); container.pack(expand=True, fill="both", padx=20, pady=10)
        for c in (0, 1, 2): container.grid_columnconfigure(c, weight=1)
        container.grid_rowconfigure(0, weight=3); container.grid_rowconfigure(1, weight=2); container.grid_rowconfigure(2, weight=0)

        # RING
        self.gauge_cpu  = RingGauge(container, vmax=100, label_scale=0.06, label_offset=1.1)
        self.gauge_mobo = RingGauge(container, vmax=95,  label_scale=0.06, label_offset=1.1)
        self.gauge_gpu  = RingGauge(container, vmax=110, label_scale=0.06, label_offset=1.1)
        self.gauge_cpu.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.gauge_mobo.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.gauge_gpu.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        # SPARK
        self.graph_cpu  = TrendGraph(container, max_seconds=HISTORY_SECONDS, max_value=110.0)
        self.graph_mobo = TrendGraph(container, max_seconds=HISTORY_SECONDS, max_value=100.0)
        self.graph_gpu  = TrendGraph(container, max_seconds=HISTORY_SECONDS, max_value=115.0)
        self.graph_cpu.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2,10))
        self.graph_mobo.grid(row=1, column=1, sticky="nsew", padx=10, pady=(2,10))
        self.graph_gpu.grid(row=1, column=2, sticky="nsew", padx=10, pady=(2,10))

        # BARS
        self.cpu_usage_bar = LoadBar(container, height=54, label_text="CPU Usage")
        self.ram_usage_bar = LoadBar(container, height=54, label_text="RAM Usage")
        self.gpu_usage_bar = LoadBar(container, height=54, label_text="GPU Usage")
        self.cpu_usage_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,6))
        self.ram_usage_bar.grid(row=2, column=1, sticky="ew", padx=10, pady=(0,6))
        self.gpu_usage_bar.grid(row=2, column=2, sticky="ew", padx=10, pady=(0,6))

        # Stato HWiNFO
        self.k32 = self.hmap = self.view = None
        self.sensors = self.entries = None
        self.cpu_key = self.mobo_key = self.gpu_key = None
        self.cpu_usage_key = self.gpu_usage_key = None
        # RAM keys (italiano + inglese)
        self.ram_used_key = self.ram_total_key = self.ram_avail_key = None
        self.ram_usage_key_percent = None

        self.cpu_disp = self.mobo_disp = self.gpu_disp = ""

        self._open_and_map(); self._find_keys()
        self.after(int(POLL_INTERVAL * 1000), self._tick)

    # ---- HWiNFO lifecycle ----
    def _open_and_map(self):
        try:
            if self.view: return
            self.k32, self.hmap, self.view = open_mapping()
            self.sensors, self.entries = load_tables(self.view)
        except Exception as e:
            self.clock_lbl.configure(text=str(e)[:120], fg="#ff7a1a")
            self.k32 = self.hmap = self.view = None

    def _reload_sensors(self):
        self._close_mapping(); self._open_and_map(); self._find_keys()

    def _close_mapping(self):
        try:
            if self.k32 and (self.hmap or self.view): close_mapping(self.k32, self.hmap, self.view)
        finally:
            self.k32 = self.hmap = self.view = None

    def _find_keys(self):
        if not (self.sensors and self.entries): return
        # Temp CPU (Enhanced → Package)
        self.cpu_key, _, _ = find_temp_entry(self.sensors, self.entries,
            label_needles=["package", "package cpu"], sensor_needles=["enhanced"])
        # MOBO (PCH / motherboard)
        self.mobo_key, self.mobo_name, _ = find_temp_entry(self.sensors, self.entries,
            label_needles=["pch temperatura","pch temperature","pch temp","pch","temperatura scheda madre","motherboard","system"],
            sensor_needles=["pch","motherboard","system","msi","asus","intel pch","z7"])
        # GPU temp
        self.gpu_key, _, _ = find_temp_entry(self.sensors, self.entries,
            label_needles=["gpu temperature","temperatura gpu","hot spot","edge temperature"],
            sensor_needles=["geforce","nvidia","radeon","amd","gpu"])

        # CPU usage: Total CPU Utility (fallback su voci simili)
        self.cpu_usage_key, _, _ = find_entry_by_exact_label_dual(self.sensors, self.entries,
            {_norm("total cpu utility")})
        if not self.cpu_usage_key:
            self.cpu_usage_key, _, _ = find_entry_fallback_with_second_if_dup(
                self.sensors, self.entries,
                label_needles=["uso totale cpu","total cpu usage","cpu usage","utilizzo cpu","cpu utility"],
                sensor_needles=["cpu","core","intel","ryzen"]
            )

        # GPU usage: carico core GPU (fallback GPU core load)
        self.gpu_usage_key, _, _ = find_entry_by_exact_label_dual(self.sensors, self.entries,
            {_norm("carico core gpu"), _norm("gpu core load")})
        if not self.gpu_usage_key:
            self.gpu_usage_key, _, _ = find_entry_fallback_with_second_if_dup(
                self.sensors, self.entries,
                label_needles=["carico core gpu","gpu core load","gpu load","gpu usage","utilizzo gpu"],
                sensor_needles=["geforce","nvidia","radeon","amd","gpu"]
            )

        # RAM usage: **Memoria fisica usata** (+ totale o disponibile)
        self.ram_used_key, _, _ = find_entry_by_exact_label_dual(self.sensors, self.entries,
            {_norm("memoria fisica usata"), _norm("physical memory used")})
        # totale (se esiste) o disponibile
        self.ram_total_key, _, _ = find_entry_by_exact_label_dual(self.sensors, self.entries,
            {_norm("memoria fisica totale"), _norm("total physical memory")})
        self.ram_avail_key, _, _ = find_entry_by_exact_label_dual(self.sensors, self.entries,
            {_norm("memoria fisica disponibile"), _norm("available physical memory")})
        # percentuale diretta (fallback)
        self.ram_usage_key_percent, _, _ = find_entry_by_exact_label_dual(self.sensors, self.entries,
            {_norm("carico memoria fisica"), _norm("memory load"), _norm("memory usage"), _norm("utilizzo memoria")})

        # label ring finali (mobo ripulita)
        self.cpu_disp  = get_cpu_name_reg() or "CPU"
        self.mobo_disp = clean_mobo_label(self.mobo_name)
        self.gpu_disp  = get_gpu_name_reg_discrete() or "GPU"

    # ---- UI helpers ----
    def _toggle_fullscreen(self):
        cur = self.overrideredirect(); self.overrideredirect(not cur); self.update_idletasks()
    def _quit(self):
        self._close_mapping(); self.destroy()

    # ---- main loop ----
    def _tick(self):
        self.clock_lbl.configure(text=f"{datetime.now():%H:%M:%S   %d.%m.%Y}", fg=FG)
        if not self.view: self._open_and_map()

        # Temp (ring + spark)
        for key, disp, gauge, graph in [
            (self.cpu_key,  self.cpu_disp,  self.gauge_cpu,  self.graph_cpu),
            (self.mobo_key, self.mobo_disp, self.gauge_mobo, self.graph_mobo),
            (self.gpu_key,  self.gpu_disp,  self.gauge_gpu,  self.graph_gpu),
        ]:
            val, _ = (read_by_key(self.view, key) if (self.view and key) else (None, None))
            gauge.set(val, disp); graph.push(val)

        # CPU usage
        cpu_use, _ = (read_by_key(self.view, self.cpu_usage_key) if (self.view and self.cpu_usage_key) else (None, None))
        self.cpu_usage_bar.set(cpu_use)

        # RAM usage (%)
        ram_percent = None
        if self.ram_used_key:
            used, _ = read_by_key(self.view, self.ram_used_key)
            total = None; avail = None
            if self.ram_total_key: total, _ = read_by_key(self.view, self.ram_total_key)
            if self.ram_avail_key: avail, _ = read_by_key(self.view, self.ram_avail_key)
            try:
                if used is not None:
                    if total and float(total) > 0:
                        ram_percent = max(0.0, min(100.0, (float(used) / float(total)) * 100.0))
                    elif avail is not None and (float(used) + float(avail)) > 0:
                        tot = float(used) + float(avail)
                        ram_percent = max(0.0, min(100.0, (float(used) / tot) * 100.0))
            except Exception:
                ram_percent = None
        if ram_percent is None and self.ram_usage_key_percent:
            ram_percent, _ = read_by_key(self.view, self.ram_usage_key_percent)
        self.ram_usage_bar.set(ram_percent)

        # GPU usage
        gpu_use, _ = (read_by_key(self.view, self.gpu_usage_key) if (self.view and self.gpu_usage_key) else (None, None))
        self.gpu_usage_bar.set(gpu_use)

        self.after(int(POLL_INTERVAL * 1000), self._tick)

if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
