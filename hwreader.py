"""
HwinfoReader: pure reader for HWiNFO Shared Memory (Windows-only), no UI, no threads.
Adds support for FAN speeds (CPU / GPU / Pump / system fans) in RPM.
"""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes
import platform
import re
import time
import winreg
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ============== SHM consts ==============
FILE_MAP_READ = 0x0004
MAP_NAMES = [
    "Global\\HWiNFO_SENS_SM2", "Global\\HWiNFO64_SENS_SM2",
    "HWiNFO_SENS_SM2", "HWiNFO64_SENS_SM2",
]
MAGIC_LITTLE = int.from_bytes(b"SiWH", "little")
MAGIC_BIG    = int.from_bytes(b"SiWH", "big")

# ============== ctypes structs ==============
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

# ============== helpers ==============
def _k32():
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

_defrag_re = re.compile(r"\s{2,}")
_rm_marks_re = re.compile(r"\(R\)|\(TM\)|\(tm\)|\(r\)|\(C\)")

def _cleanup_name(s: str) -> str:
    s = _rm_marks_re.sub("", s or "")
    s = _defrag_re.sub(" ", s).strip(" -_/")
    return s

def _cstr(b: bytes) -> str:
    i = b.find(b"\x00")
    if i != -1:
        b = b[:i]
    try:
        return b.decode("utf-8", "ignore")
    except Exception:
        return ""

# ============== map / unmap ==============
def _open_mapping():
    if platform.system() != "Windows":
        raise RuntimeError("Richiede Windows")
    k = _k32()
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
    raise RuntimeError("HWiNFO Shared Memory non trovata (abilita 'Shared Memory Support' e apri la finestra Sensors).")

def _close_mapping(k, h, view):
    try:
        if view:
            k.UnmapViewOfFile(view)
    except Exception:
        pass
    try:
        if h:
            k.CloseHandle(h)
    except Exception:
        pass

# ============== table loaders ==============
def _load_tables(view) -> Tuple[List[dict], List[dict]]:
    base = int(view.value)
    hdr = Hdr.from_address(base)
    if hdr.magic not in (MAGIC_LITTLE, MAGIC_BIG):
        raise RuntimeError(f"Header non riconosciuto (magic=0x{hdr.magic:08X})")
    pS = base + hdr.sensor_section_offset
    sensors: List[dict] = []
    for i in range(hdr.sensor_element_count):
        s = Sensor.from_address(pS + i * hdr.sensor_element_size)
        sensors.append({
            "id": s.id, "instance": s.instance,
            "name_user": _cstr(bytes(s.name_user)).strip(),
            "name_original": _cstr(bytes(s.name_original)).strip(),
        })
    pE = base + hdr.entry_section_offset
    entries: List[dict] = []
    for i in range(hdr.entry_element_count):
        e = Entry.from_address(pE + i * hdr.entry_element_size)
        entries.append({
            "type": e.type, "sensor_index": e.sensor_index, "id": e.id,
            "label_user": _cstr(bytes(e.name_user)).strip(),
            "label_original": _cstr(bytes(e.name_original)).strip(),
            "unit": _cstr(bytes(e.unit)).strip(),
            "value": float(e.value),
        })
    return sensors, entries

# ============== discovery helpers ==============
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _match_any(needles: Sequence[str], hay: str) -> bool:
    h = (hay or "").lower()
    return any(n and n.lower() in h for n in needles)

def _sensor_name(sensors: List[dict], idx: int) -> str:
    if 0 <= idx < len(sensors):
        s = sensors[idx]
        return s.get("name_user") or s.get("name_original") or ""
    return ""

def _find_temp_entry(sensors, entries, *, label_needles, sensor_needles):
    for e in entries:
        lbl = (e["label_user"] or e["label_original"])
        sname = _sensor_name(sensors, e["sensor_index"])
        if _match_any(label_needles, lbl) and _match_any(sensor_needles, sname):
            return ((e["sensor_index"], e["id"]), sname, lbl)
    return (None, "", "")

def _find_entry_by_exact_label_dual(sensors, entries, targets_norm: set[str]):
    for e in entries:
        if _norm(e["label_user"]) in targets_norm or _norm(e["label_original"]) in targets_norm:
            sname = _sensor_name(sensors, e["sensor_index"])
            return ((e["sensor_index"], e["id"]), sname, e["label_user"] or e["label_original"])
    return (None, "", "")

def _find_entry_fallback_with_second_if_dup(sensors, entries, *, label_needles, sensor_needles):
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

# ============== registry labels ==============
def clean_mobo_label(raw: str) -> str:
    s = raw or ""
    s = re.sub(r"^\s*Sistema:\s*", "", s, flags=re.I)
    s = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s)
    s = _defrag_re.sub(" ", s).strip(" -_/")
    return s or "Motherboard / PCH"

def get_cpu_name_reg() -> Optional[str]:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
            return _cleanup_name(val)
    except OSError:
        return None

def get_gpu_name_reg_discrete() -> Optional[str]:
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    dgpu = []; igpu = []; others = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            i = 0
            while True:
                try:
                    subname = winreg.EnumKey(root, i); i += 1
                except OSError:
                    break
                if not re.fullmatch(r"\d{4}", subname):
                    continue
                try:
                    with winreg.OpenKey(root, subname) as k:
                        desc, _ = winreg.QueryValueEx(k, "DriverDesc")
                        name = _cleanup_name(desc)
                        prov = ""
                        try:
                            prov, _ = winreg.QueryValueEx(k, "ProviderName")
                        except OSError:
                            pass
                        src = (str(prov) + " " + name).lower()
                        if re.search(r"nvidia|geforce|amd|radeon", src):
                            dgpu.append(name)
                        elif re.search(r"intel", src):
                            igpu.append(name)
                        else:
                            others.append(name)
                except OSError:
                    continue
    except OSError:
        return None
    return (dgpu[0] if dgpu else (igpu[0] if igpu else (others[0] if others else None)))

# ============== read by key ==============
def _read_by_key(view, key: Optional[Tuple[int,int]]):
    if not key:
        return None, None
    base = int(view.value)
    hdr = Hdr.from_address(base)
    pE = base + hdr.entry_section_offset
    for i in range(hdr.entry_element_count):
        e = Entry.from_address(pE + i * hdr.entry_element_size)
        if e.sensor_index == key[0] and e.id == key[1]:
            return float(e.value), (_cstr(bytes(e.name_user)) or _cstr(bytes(e.name_original)))
    return None, None

# ============== reader class ==============
@dataclass
class MetricsKeys:
    cpu_temp: Optional[Tuple[int,int]] = None
    pch_temp: Optional[Tuple[int,int]] = None
    gpu_temp: Optional[Tuple[int,int]] = None
    cpu_usage: Optional[Tuple[int,int]] = None
    gpu_usage: Optional[Tuple[int,int]] = None
    ram_used: Optional[Tuple[int,int]] = None
    ram_total: Optional[Tuple[int,int]] = None
    ram_avail: Optional[Tuple[int,int]] = None
    ram_usage_percent: Optional[Tuple[int,int]] = None
    # Fans
    fan_cpu: Optional[Tuple[int,int]] = None
    fan_gpu: Optional[Tuple[int,int]] = None
    fan_pump: Optional[Tuple[int,int]] = None
    fan_sys: List[Tuple[str, Tuple[int,int]]] = None  # list of (label, key)

class HwinfoReader:
    def __init__(self):
        self.k32 = None
        self.hmap = None
        self.view = None
        self.sensors: Optional[List[dict]] = None
        self.entries: Optional[List[dict]] = None
        self.keys = MetricsKeys()
        self._labels: Dict[str, str] = {"cpu": "CPU", "mobo": "Motherboard / PCH", "gpu": "GPU"}
        self._mobo_sensor_name: str = ""

    # --- lifecycle ---
    def open(self):
        self.k32, self.hmap, self.view = _open_mapping()

    def close(self):
        if self.k32 and (self.hmap or self.view):
            _close_mapping(self.k32, self.hmap, self.view)
        self.k32 = self.hmap = self.view = None

    def scan(self):
        """Load tables and discover keys + labels + fans."""
        if not self.view:
            self.open()
        self.sensors, self.entries = _load_tables(self.view)

        # temps
        self.keys.cpu_temp, _, _ = _find_temp_entry(
            self.sensors, self.entries,
            label_needles=["package", "package cpu", "cpu"], sensor_needles=["enhanced"]
        )
        pch_key, mobo_name, _ = _find_temp_entry(
            self.sensors, self.entries,
            label_needles=["pch temperatura", "pch temperature", "pch temp", "pch", "temperatura scheda madre", "motherboard", "system", "scheda madre"],
            sensor_needles=["pch", "motherboard", "system", "msi", "asus", "intel pch", "z7"]
        )
        self.keys.pch_temp = pch_key
        self._mobo_sensor_name = mobo_name
        self.keys.gpu_temp, _, _ = _find_temp_entry(
            self.sensors, self.entries,
            label_needles=["gpu temperature", "temperatura gpu", "hot spot", "edge temperature"],
            sensor_needles=["geforce", "nvidia", "radeon", "amd", "gpu"]
        )

        # usage
        self.keys.cpu_usage, _, _ = _find_entry_by_exact_label_dual(self.sensors, self.entries, { _norm("total cpu utility") })
        if not self.keys.cpu_usage:
            self.keys.cpu_usage, _, _ = _find_entry_fallback_with_second_if_dup(
                self.sensors, self.entries,
                label_needles=["uso totale cpu", "total cpu usage", "cpu usage", "utilizzo cpu", "cpu utility"],
                sensor_needles=["cpu", "core", "intel", "ryzen"]
            )
        self.keys.gpu_usage, _, _ = _find_entry_by_exact_label_dual(self.sensors, self.entries, { _norm("carico core gpu"), _norm("gpu core load") })
        if not self.keys.gpu_usage:
            self.keys.gpu_usage, _, _ = _find_entry_fallback_with_second_if_dup(
                self.sensors, self.entries,
                label_needles=["carico core gpu", "gpu core load", "gpu load", "gpu usage", "utilizzo gpu", "uso gpu"],
                sensor_needles=["geforce", "nvidia", "radeon", "amd", "gpu"]
            )

        # RAM
        self.keys.ram_used, _, _ = _find_entry_by_exact_label_dual(self.sensors, self.entries, { _norm("memoria fisica usata"), _norm("physical memory used") })
        self.keys.ram_total, _, _ = _find_entry_by_exact_label_dual(self.sensors, self.entries, { _norm("memoria fisica totale"), _norm("total physical memory") })
        self.keys.ram_avail, _, _ = _find_entry_by_exact_label_dual(self.sensors, self.entries, { _norm("memoria fisica disponibile"), _norm("available physical memory") })
        self.keys.ram_usage_percent, _, _ = _find_entry_by_exact_label_dual(self.sensors, self.entries, { _norm("carico memoria fisica"), _norm("memory load"), _norm("memory usage"), _norm("utilizzo memoria") })

        # labels
        self._labels["cpu"] = get_cpu_name_reg() or "CPU"
        self._labels["mobo"] = clean_mobo_label(self._mobo_sensor_name)
        self._labels["gpu"] = get_gpu_name_reg_discrete() or "GPU"

        # Fans discovery
        self._discover_fans()

    def _discover_fans(self):
        self.keys.fan_cpu = None
        self.keys.fan_gpu = None
        self.keys.fan_pump = None
        self.keys.fan_sys = []
        entries = self.entries or []
        sensors = self.sensors or []

        def is_rpm_unit(e):
            u = (e.get("unit") or "").strip().lower()
            return u in ("rpm", "r.p.m.")

        for e in entries:
            if not is_rpm_unit(e):
                continue

            lbl = (e.get("label_user") or e.get("label_original") or "").strip()
            nlbl = _norm(lbl)       # es: "ventola0" → "ventola0"
            sname = _sensor_name(sensors, e["sensor_index"]) or ""
            nsname = _norm(sname)
            key = (e["sensor_index"], e["id"])

            # ==============================
            #  PRIORITÀ: Ventola0 / Ventola1
            # ==============================

            # Ventola0 → CPU
            if nlbl == "ventola0":
                self.keys.fan_cpu = key
                continue

            # Ventola1 → Pump
            if nlbl == "ventola1":
                self.keys.fan_pump = key
                continue

            # Le altre ventole (Ventola2, Ventola5, ecc.) → system fans
            if nlbl.startswith("ventola"):
                self.keys.fan_sys.append((lbl, key))
                continue

            # ==============================
            #  Regole generiche
            # ==============================

            # Pump
            if any(tok in nlbl for tok in ("pump", "pompa", "aio pump", "water pump", "pump1", "pump 1")):
                if self.keys.fan_pump is None:
                    self.keys.fan_pump = key
                continue

            # CPU fan generico
            if (nlbl == "cpu") or ("cpu" in nlbl and "fan" in nlbl) or ("cpu" in nsname and "fan" in nlbl):
                if self.keys.fan_cpu is None:
                    self.keys.fan_cpu = key
                continue

            # GPU fan
            if ("gpu" in nlbl or "gpu" in nsname) and ("fan" in nlbl or "ventola" in nlbl or "fan" in nsname):
                if self.keys.fan_gpu is None:
                    self.keys.fan_gpu = key
                continue

            # System / chassis fans
            if any(tok in nlbl for tok in ("sistema", "system", "sys fan", "chassis", "case fan", "pch fan")) or \
               re.search(r"(sistema|system)\s*\d+", nlbl):
                disp = lbl or f"Fan {e['id']}"
                self.keys.fan_sys.append((disp, key))
                continue

            # Generic motherboard/system fan
            if "fan" in nlbl and any(t in nsname for t in ("motherboard", "system", "pch", "asus", "msi", "gigabyte")):
                disp = lbl or f"Fan {e['id']}"
                self.keys.fan_sys.append((disp, key))

    # --- snapshot ---
    def read_snapshot(self) -> Dict:
        cpu_t = pch_t = gpu_t = None
        cpu_use = ram_pct = gpu_use = None
        fan_cpu = fan_gpu = pump_rpm = None
        fans_sys: Dict[str, Optional[float]] = {}

        if self.view and self.keys:
            try:
                cpu_t, _ = _read_by_key(self.view, self.keys.cpu_temp)
                pch_t, _ = _read_by_key(self.view, self.keys.pch_temp)
                gpu_t, _ = _read_by_key(self.view, self.keys.gpu_temp)
                cpu_use, _ = _read_by_key(self.view, self.keys.cpu_usage)
                gpu_use, _ = _read_by_key(self.view, self.keys.gpu_usage)

                # RAM percent
                used, _ = _read_by_key(self.view, self.keys.ram_used)
                total = avail = None
                if self.keys.ram_total:
                    total, _ = _read_by_key(self.view, self.keys.ram_total)
                if self.keys.ram_avail:
                    avail, _ = _read_by_key(self.view, self.keys.ram_avail)
                try:
                    if used is not None:
                        if total and float(total) > 0:
                            ram_pct = max(0.0, min(100.0, (float(used) / float(total)) * 100.0))
                        elif avail is not None and (float(used) + float(avail)) > 0:
                            tot = float(used) + float(avail)
                            ram_pct = max(0.0, min(100.0, (float(used) / tot) * 100.0))
                except Exception:
                    ram_pct = None
                if ram_pct is None and self.keys.ram_usage_percent:
                    ram_pct, _ = _read_by_key(self.view, self.keys.ram_usage_percent)

                # Fans
                fan_cpu, _ = _read_by_key(self.view, self.keys.fan_cpu)
                fan_gpu, _ = _read_by_key(self.view, self.keys.fan_gpu)
                if self.keys.fan_pump:
                    pump_rpm, _ = _read_by_key(self.view, self.keys.fan_pump)
                for disp, k in (self.keys.fan_sys or []):
                    val, _ = _read_by_key(self.view, k)
                    fans_sys[disp] = val

            except Exception:
                pass

        return {
            'timestamp': time.time(),
            'cpu_temp': cpu_t,
            'pch_temp': pch_t,
            'gpu_temp': gpu_t,
            'cpu_usage': cpu_use,
            'ram_usage': ram_pct,
            'gpu_usage': gpu_use,
            'labels': dict(self._labels),
            'fan_cpu_rpm': fan_cpu,
            'fan_gpu_rpm': fan_gpu,
            'fan_pump_rpm': pump_rpm,
            'fans': { 'sys': fans_sys } if fans_sys else {},
        }

    # --- utility ---
    def labels(self) -> Dict[str, str]:
        return dict(self._labels)
