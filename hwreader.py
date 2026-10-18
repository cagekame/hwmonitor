"""
HwinfoReader: pure reader for HWiNFO Shared Memory (Windows-only), no UI, no threads.
- scan():          discovers which entries to use (config mapping first, heuristics otherwise)
- read_snapshot(): returns a model.Snapshot
- list_entries():  every entry with a given unit (used by the settings window)
"""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes
import platform
import re
import time
import winreg
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from model import Snapshot

Key = Tuple[int, int]  # (sensor_index, entry_id)

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

# ============== read values ==============
def _read_values(view, wanted: set) -> Dict[Key, float]:
    """One pass over the entry table → {key: value} for the wanted keys."""
    out: Dict[Key, float] = {}
    if not wanted:
        return out
    base = int(view.value)
    hdr = Hdr.from_address(base)
    pE = base + hdr.entry_section_offset
    for i in range(hdr.entry_element_count):
        e = Entry.from_address(pE + i * hdr.entry_element_size)
        k = (e.sensor_index, e.id)
        if k in wanted:
            out[k] = float(e.value)
            if len(out) == len(wanted):
                break
    return out

# ============== reader class ==============
@dataclass
class MetricsKeys:
    cpu_temp: Optional[Key] = None
    mobo_temp: Optional[Key] = None
    gpu_temp: Optional[Key] = None
    cpu_usage: Optional[Key] = None
    gpu_usage: Optional[Key] = None
    ram_used: Optional[Key] = None
    ram_total: Optional[Key] = None
    ram_avail: Optional[Key] = None
    ram_usage_percent: Optional[Key] = None
    # Fans
    fan_cpu: Optional[Key] = None
    fan_gpu: Optional[Key] = None
    fan_pump: Optional[Key] = None
    fan_sys: List[Tuple[str, Key]] = field(default_factory=list)  # (label, key)

    def all(self) -> set:
        keys = {getattr(self, n) for n in ("cpu_temp", "mobo_temp", "gpu_temp", "cpu_usage", "gpu_usage", "ram_used",
                                           "ram_total", "ram_avail", "ram_usage_percent", "fan_cpu", "fan_gpu", "fan_pump")}
        keys.update(k for _, k in self.fan_sys)
        keys.discard(None)
        return keys

class HwinfoReader:
    def __init__(self, config: dict = None):
        self.k32 = None
        self.hmap = None
        self.view = None
        self.sensors: Optional[List[dict]] = None
        self.entries: Optional[List[dict]] = None
        self.keys = MetricsKeys()
        self._labels: Dict[str, str] = {"cpu": "CPU", "mobo": "Motherboard / PCH", "gpu": "GPU"}
        self._mobo_sensor_name: str = ""
        self.config = config or {}

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

        # Check for custom sensor mapping
        sensor_mapping = self.config.get("sensor_mapping", {})
        
        # CPU temp - use custom if available
        cpu_map = sensor_mapping.get("cpu", {})
        if cpu_map and cpu_map.get("sensor_index") is not None and cpu_map.get("entry_id") is not None:
            self.keys.cpu_temp = (cpu_map["sensor_index"], cpu_map["entry_id"])
        else:
            self.keys.cpu_temp, _, _ = _find_temp_entry(
                self.sensors, self.entries,
                label_needles=["package", "package cpu", "cpu"], sensor_needles=["enhanced"]
            )
        
        # Mobo/PCH temp - use custom if available
        mobo_map = sensor_mapping.get("mobo", {})
        if mobo_map and mobo_map.get("sensor_index") is not None and mobo_map.get("entry_id") is not None:
            mobo_key = (mobo_map["sensor_index"], mobo_map["entry_id"])
            mobo_name = mobo_map.get("label", "Motherboard")
        else:
            mobo_key, mobo_name, _ = _find_temp_entry(
                self.sensors, self.entries,
                label_needles=["pch temperatura", "pch temperature", "pch temp", "pch", "temperatura scheda madre", "motherboard", "system", "scheda madre"],
                sensor_needles=["pch", "motherboard", "system", "msi", "asus", "intel pch", "z7"]
            )
        self.keys.mobo_temp = mobo_key
        self._mobo_sensor_name = mobo_name
        
        # GPU temp - use custom if available
        gpu_map = sensor_mapping.get("gpu", {})
        if gpu_map and gpu_map.get("sensor_index") is not None and gpu_map.get("entry_id") is not None:
            self.keys.gpu_temp = (gpu_map["sensor_index"], gpu_map["entry_id"])
        else:
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

        # labels - get actual sensor names
        # Helper to get sensor name from key
        def get_sensor_name_from_key(key):
            if not key:
                return None
            sensor_idx, _ = key
            if 0 <= sensor_idx < len(self.sensors):
                s = self.sensors[sensor_idx]
                name = s.get("name_user") or s.get("name_original") or ""
                # Extract middle part between colons: "string" : "MIDDLE" : "string"
                if ":" in name:
                    parts = name.split(":")
                    if len(parts) >= 2:
                        # Take the middle part (or first part if only 2 parts)
                        middle_idx = 1 if len(parts) >= 3 else 0
                        return parts[middle_idx].strip()
                return name
            return None
        
        # CPU label - try to get from actual sensor used
        cpu_sensor_name = get_sensor_name_from_key(self.keys.cpu_temp)
        if cpu_sensor_name:
            self._labels["cpu"] = cpu_sensor_name
        else:
            self._labels["cpu"] = get_cpu_name_reg() or "CPU"
        
        # Mobo label
        mobo_sensor_name = get_sensor_name_from_key(self.keys.mobo_temp)
        if mobo_sensor_name:
            self._labels["mobo"] = clean_mobo_label(mobo_sensor_name)
        else:
            self._labels["mobo"] = clean_mobo_label(self._mobo_sensor_name)
        
        # GPU label
        gpu_sensor_name = get_sensor_name_from_key(self.keys.gpu_temp)
        if gpu_sensor_name:
            self._labels["gpu"] = gpu_sensor_name
        else:
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
        rpm_entries = [e for e in entries if (e.get("unit") or "").strip().lower() in ("rpm", "r.p.m.")]

        # 1) explicit mapping from config: {"cpu"|"gpu"|"pump": {"sensor_index", "entry_id"} or {"label"}}
        fan_mapping = self.config.get("fan_mapping", {}) or {}
        attr = {"cpu": "fan_cpu", "gpu": "fan_gpu", "pump": "fan_pump"}
        mapped: set = set()
        for role, m in fan_mapping.items():
            if role not in attr or not isinstance(m, dict):
                continue
            key = None
            if m.get("sensor_index") is not None and m.get("entry_id") is not None:
                key = (m["sensor_index"], m["entry_id"])
            elif m.get("label"):
                want = _norm(m["label"])
                for e in rpm_entries:
                    if _norm(e.get("label_user") or e.get("label_original") or "") == want:
                        key = (e["sensor_index"], e["id"])
                        break
            if key:
                setattr(self.keys, attr[role], key)
                mapped.add(key)

        # 2) heuristics for everything else
        for e in rpm_entries:
            lbl = (e.get("label_user") or e.get("label_original") or "").strip()
            nlbl = _norm(lbl)       # es: "ventola0" → "ventola0"
            sname = _sensor_name(sensors, e["sensor_index"]) or ""
            nsname = _norm(sname)
            key = (e["sensor_index"], e["id"])
            if key in mapped:
                continue

            # Mobo headers named "VentolaN" (Italian HWiNFO) → system fans
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
    def read_snapshot(self) -> Snapshot:
        snap = Snapshot(labels=dict(self._labels))
        if not self.view:
            return snap
        k = self.keys
        try:
            vals = _read_values(self.view, k.all())
        except Exception:
            return snap
        get = lambda key: vals.get(key) if key else None

        snap.cpu_temp = get(k.cpu_temp)
        snap.mobo_temp = get(k.mobo_temp)
        snap.gpu_temp = get(k.gpu_temp)
        snap.cpu_usage = get(k.cpu_usage)
        snap.gpu_usage = get(k.gpu_usage)

        # RAM (HWiNFO reports MB)
        used, total, avail = get(k.ram_used), get(k.ram_total), get(k.ram_avail)
        if used is not None:
            if total and total > 0:
                snap.ram_total_mb = total
            elif avail is not None:
                snap.ram_total_mb = used + avail
            snap.ram_used_mb = used
            if avail is not None:
                snap.ram_free_mb = avail
            elif snap.ram_total_mb is not None:
                snap.ram_free_mb = max(0.0, snap.ram_total_mb - used)
            if snap.ram_total_mb and snap.ram_total_mb > 0:
                snap.ram_usage = max(0.0, min(100.0, used / snap.ram_total_mb * 100.0))
        if snap.ram_usage is None:
            snap.ram_usage = get(k.ram_usage_percent)

        # Fans
        snap.fan_cpu_rpm = get(k.fan_cpu)
        snap.fan_gpu_rpm = get(k.fan_gpu)
        snap.fan_pump_rpm = get(k.fan_pump)
        snap.fans_sys = {disp: get(key) for disp, key in k.fan_sys}
        return snap

    def data_age(self) -> Optional[float]:
        """Seconds since HWiNFO last refreshed the shared memory (header `last_update`, Unix time), or None.
        The free HWiNFO stops refreshing after 12h while the mapping stays open → large age = stale data."""
        if not self.view:
            return None
        try:
            lu = Hdr.from_address(int(self.view.value)).last_update
        except Exception:
            return None
        return max(0.0, time.time() - lu) if lu > 0 else None

    # --- utility ---
    def labels(self) -> Dict[str, str]:
        return dict(self._labels)

    def list_entries(self, units: Sequence[str]) -> List[dict]:
        """All entries whose unit is in `units` (case-insensitive), from the last scan():
        [{sensor_index, entry_id, label, sensor, value, unit}]"""
        want = {u.lower() for u in units}
        out = []
        for e in self.entries or []:
            unit = (e.get("unit") or "").strip()
            if unit.lower() not in want:
                continue
            out.append({
                "sensor_index": e["sensor_index"], "entry_id": e["id"],
                "label": e.get("label_user") or e.get("label_original") or "",
                "sensor": _sensor_name(self.sensors or [], e["sensor_index"]),
                "value": e.get("value"), "unit": unit,
            })
        return out