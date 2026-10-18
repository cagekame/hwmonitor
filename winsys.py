"""
Windows system helpers: monitors, display mode, uptime, hostname.
Safe no-ops / fallbacks on other platforms.
"""

from __future__ import annotations

import os
import platform
from typing import List, Optional, Tuple

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
    _gdi = ct.windll.gdi32
    _k32 = ct.windll.kernel32
    _u32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ct.POINTER(RECT), PMONITORENUMPROC, wintypes.LPARAM]
    _u32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ct.POINTER(MONITORINFOEXW)]
    _u32.MonitorFromWindow.restype = wintypes.HMONITOR
    _u32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _gdi.CreateDCW.restype = wintypes.HDC
    _gdi.CreateDCW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, ct.c_void_p]
    _gdi.GetDeviceCaps.argtypes = [wintypes.HDC, ct.c_int]
    _gdi.DeleteDC.argtypes = [wintypes.HDC]
    _k32.GetTickCount64.restype = ct.c_ulonglong

    class PROCESSENTRY32W(ct.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ct.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ct.c_long), ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32FirstW.argtypes = [wintypes.HANDLE, ct.POINTER(PROCESSENTRY32W)]
    _k32.Process32NextW.argtypes = [wintypes.HANDLE, ct.POINTER(PROCESSENTRY32W)]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ct.POINTER(wintypes.FILETIME)] * 4
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _WIN = True
except Exception:
    _WIN = False


def list_monitors() -> List[dict]:
    """[{device, x, y, w, h, primary}] in system enumeration order."""
    if not _WIN:
        import tkinter as tk
        root = tk.Tk()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return [{"device": "primary", "x": 0, "y": 0, "w": w, "h": h, "primary": True}]
    mons: List[dict] = []

    def _cb(hMon, hdc, lprc, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ct.sizeof(MONITORINFOEXW)
        if _u32.GetMonitorInfoW(hMon, ct.byref(info)):
            r = info.rcMonitor
            mons.append({"device": info.szDevice, "x": r.left, "y": r.top, "w": r.right - r.left,
                         "h": r.bottom - r.top, "primary": bool(info.dwFlags & 1)})
        return True

    _u32.EnumDisplayMonitors(0, None, PMONITORENUMPROC(_cb), 0)
    return mons


def pick_monitor(cfg_monitor: dict) -> dict:
    """Choose a monitor by position (left/right/top/bottom/primary) or index."""
    mons = list_monitors()
    if not mons:
        return {"x": 0, "y": 0, "w": 800, "h": 600}
    sel = (cfg_monitor.get("select") or "primary").lower()
    if sel == "primary":
        return next((m for m in mons if m.get("primary")), mons[0])
    if sel == "left":
        return min(mons, key=lambda m: (m["x"], m["y"]))
    if sel == "right":
        return max(mons, key=lambda m: (m["x"], m["y"]))
    if sel == "top":
        return min(mons, key=lambda m: (m["y"], m["x"]))
    if sel == "bottom":
        return max(mons, key=lambda m: (m["y"], m["x"]))
    if sel == "index":
        idx = int(cfg_monitor.get("index", 1)) - int(cfg_monitor.get("index_base", 1))
        return mons[idx] if 0 <= idx < len(mons) else mons[-1]
    return mons[0]


def display_info(hwnd: int) -> Optional[Tuple[int, int, int]]:
    """(width, height, refresh_hz) of the monitor hosting `hwnd`, or None."""
    if not _WIN:
        return None
    try:
        hmon = _u32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = MONITORINFOEXW()
        info.cbSize = ct.sizeof(MONITORINFOEXW)
        if not _u32.GetMonitorInfoW(hmon, ct.byref(info)):
            return None
        hdc = _gdi.CreateDCW(info.szDevice, None, None, None)
        if not hdc:
            return None
        try:
            # DESKTOPHORZRES=118, DESKTOPVERTRES=117, VREFRESH=116
            return _gdi.GetDeviceCaps(hdc, 118), _gdi.GetDeviceCaps(hdc, 117), _gdi.GetDeviceCaps(hdc, 116)
        finally:
            _gdi.DeleteDC(hdc)
    except Exception:
        return None


def uptime_seconds() -> Optional[float]:
    if not _WIN:
        return None
    try:
        return _k32.GetTickCount64() / 1000.0
    except Exception:
        return None


def process_start_time(exe_names) -> Optional[float]:
    """Epoch start time of the oldest running process whose exe name is in `exe_names`, or None."""
    if not _WIN:
        return None
    names = {n.lower() for n in exe_names}
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return None
    pids = []
    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ct.sizeof(PROCESSENTRY32W)
        ok = _k32.Process32FirstW(snap, ct.byref(pe))
        while ok:
            if pe.szExeFile.lower() in names:
                pids.append(pe.th32ProcessID)
            ok = _k32.Process32NextW(snap, ct.byref(pe))
    finally:
        _k32.CloseHandle(snap)
    starts = []
    for pid in pids:
        h = _k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            continue
        try:
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if _k32.GetProcessTimes(h, ct.byref(c), ct.byref(e), ct.byref(k), ct.byref(u)):
                ft = (c.dwHighDateTime << 32) | c.dwLowDateTime
                starts.append(ft / 1e7 - 11644473600.0)  # FILETIME (1601) → Unix epoch
        finally:
            _k32.CloseHandle(h)
    return min(starts) if starts else None


def hostname() -> str:
    return os.environ.get("COMPUTERNAME") or platform.node()
