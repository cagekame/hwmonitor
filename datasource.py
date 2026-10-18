"""
HwinfoPoller: drives HwinfoReader on the Tk event loop.
- polls every `poll.interval_sec` (read live from the shared config dict)
- reports connection changes; "connected" also requires fresh data (HWiNFO free stops
  refreshing the shared memory after 12h while keeping it open)
- when offline, retries every `hwinfo.retry_sec`
- rescan() re-discovers sensors (after a mapping change or on hotkey)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from config import get_path
from hwreader import HwinfoReader
from model import Snapshot


class HwinfoPoller:
    def __init__(self, tk_root, cfg: dict,
                 on_snapshot: Callable[[Snapshot], None],
                 on_status: Callable[[bool], None],
                 on_labels: Callable[[Dict[str, str]], None]):
        self.root = tk_root
        self.cfg = cfg  # shared, updated in place by the app → reader sees mapping changes
        self.on_snapshot, self.on_status, self.on_labels = on_snapshot, on_status, on_labels
        self.reader: Optional[HwinfoReader] = None
        self.connected = False
        self._tick_job = None
        self._reconnect_job = None

    # ---------- settings (read live) ----------
    def _num(self, path: str, default: float) -> float:
        try:
            return float(get_path(self.cfg, path, default))
        except (TypeError, ValueError):
            return default

    @property
    def interval_ms(self) -> int:
        return max(100, int(self._num("poll.interval_sec", 1.0) * 1000))

    @property
    def retry_ms(self) -> int:
        return max(1000, int(self._num("hwinfo.retry_sec", 10) * 1000))

    @property
    def stale_after(self) -> float:
        return max(3.0, self._num("hwinfo.stale_after_sec", 15))

    # ---------- lifecycle ----------
    def start(self):
        if not self._connect():
            self._schedule_reconnect()
        self._tick()

    def stop(self):
        for job in (self._tick_job, self._reconnect_job):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
        self._tick_job = self._reconnect_job = None
        self._close_reader()

    def rescan(self):
        if self.connected and self.reader is not None:
            try:
                self.reader.scan()
                self.on_labels(self.reader.labels())
            except Exception:
                pass

    # ---------- internals ----------
    def _set_connected(self, ok: bool):
        self.connected = ok
        self.on_status(ok)

    def _close_reader(self):
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass

    def _stale(self) -> bool:
        age = self.reader.data_age() if self.reader else None
        return age is not None and age > self.stale_after

    def _connect(self) -> bool:
        try:
            if self.reader is None:
                self.reader = HwinfoReader(self.cfg)
            self.reader.open()
            if self._stale():
                raise RuntimeError("HWiNFO shared memory is not being updated")
            self.reader.scan()
            self.on_labels(self.reader.labels())
            self._set_connected(True)
            return True
        except Exception:
            self._close_reader()
            self._set_connected(False)
            return False

    def _drop(self):
        self._close_reader()
        self._set_connected(False)
        self._schedule_reconnect()

    def _tick(self):
        try:
            if self.connected:
                if self._stale():
                    self._drop()
                else:
                    self.on_snapshot(self.reader.read_snapshot())
        except Exception:
            self._drop()
        finally:
            self._tick_job = self.root.after(self.interval_ms, self._tick)

    def _schedule_reconnect(self):
        if self._reconnect_job is None:
            self._reconnect_job = self.root.after(self.retry_ms, self._reconnect)

    def _reconnect(self):
        self._reconnect_job = None
        if not self.connected and not self._connect():
            self._schedule_reconnect()
