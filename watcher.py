# watcher_dashboard.py
import time
import subprocess
import sys
import os
import ctypes as ct
from ctypes import wintypes
import argparse
from datetime import datetime

DASHBOARD_SCRIPT = "Dashboard.py"
CHECK_EVERY_SEC = 3        # polling
STABLE_HITS = 2            # conferme consecutive prima di avviare
REQUIRED_MIN_MONITORS = 3  # di default: avvia quando ci sono almeno 3 monitor

LOGFILE = "watcher.log"

# ---------- logging ----------
def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------- enum monitor ----------
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
u32 = ct.windll.user32
u32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ct.POINTER(RECT), PMONITORENUMPROC, wintypes.LPARAM]
u32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ct.POINTER(MONITORINFOEXW)]

def list_monitors():
    mons = []
    def _cb(hMon, hdc, lprc, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ct.sizeof(MONITORINFOEXW)
        if u32.GetMonitorInfoW(hMon, ct.byref(info)):
            x, y = info.rcMonitor.left, info.rcMonitor.top
            w = info.rcMonitor.right - info.rcMonitor.left
            h = info.rcMonitor.bottom - info.rcMonitor.top
            mons.append({"device": info.szDevice, "x": x, "y": y, "w": w, "h": h})
        return True
    u32.EnumDisplayMonitors(0, None, PMONITORENUMPROC(_cb), 0)
    return mons

def monitors_ready(min_required: int) -> bool:
    mons = list_monitors()
    log(f"Monitor rilevati: {len(mons)}")
    return len(mons) >= max(1, min_required)

# ---------- single instance via Windows mutex ----------
def single_instance_mutex(name="Global\\DashboardWatcherMutex"):
    k32 = ct.windll.kernel32
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.GetLastError.restype = wintypes.DWORD
    handle = k32.CreateMutexW(None, False, name)
    already = (k32.GetLastError() == 183)  # ERROR_ALREADY_EXISTS
    return handle, already

def launch_dashboard():
    python_exe = sys.executable
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), DASHBOARD_SCRIPT)
    log(f"Avvio dashboard: {script}")
    try:
        subprocess.Popen([python_exe, script], close_fds=True)
        return True
    except Exception as e:
        log(f"ERRORE avvio dashboard: {e}")
        return False

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=REQUIRED_MIN_MONITORS,
                        help="Numero minimo di monitor per avviare (default: 3)")
    parser.add_argument("--interval", type=int, default=CHECK_EVERY_SEC,
                        help="Intervallo di polling in secondi (default: 3)")
    parser.add_argument("--hits", type=int, default=STABLE_HITS,
                        help="Conferme necessarie prima di avviare (default: 2)")
    parser.add_argument("--list", action="store_true", help="Elenca i monitor e termina")
    parser.add_argument("--force", action="store_true", help="Avvia subito la dashboard")
    args = parser.parse_args()

    if args.list:
        mons = list_monitors()
        if not mons:
            print("Nessun monitor rilevato.")
        else:
            print("Monitor rilevati:")
            for m in mons:
                print(f" - {m['device']}  ({m['w']}x{m['h']} @ {m['x']},{m['y']})")
        return

    _, already = single_instance_mutex()
    if already:
        log("Watcher già in esecuzione. Esco.")
        return

    if args.force:
        log("--force: avvio immediato richiesto")
        launch_dashboard()
        return

    log(f"Watcher attivo. min={args.min}, poll={args.interval}s, hits={args.hits}")

    stable = 0
    launched = False

    while True:
        try:
            if not launched:
                if monitors_ready(args.min):
                    stable += 1
                    log(f"Conferma #{stable}/{args.hits}")
                else:
                    stable = 0

                if stable >= args.hits:
                    if launch_dashboard():
                        launched = True
                        log("Dashboard avviata. Watcher in ascolto (non rilancia).")
        except Exception as e:
            log(f"Errore ciclo: {e}")

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
