"""
HW Monitor — entry point.
  python main.py              start the dashboard
  python main.py --settings   start and open the settings window
"""

from __future__ import annotations

import sys

import config
from datasource import HwinfoPoller
from ui.dashboard import DashboardApp


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    cfg = config.load_config()

    app = DashboardApp(cfg)
    app.place_window()

    poller = HwinfoPoller(app, cfg,
                          on_snapshot=app.post_snapshot,
                          on_status=app.set_connection_status,
                          on_labels=app.set_labels)
    app.on_rescan = poller.rescan
    app.add_close_handler(poller.stop)
    poller.start()

    if "--settings" in argv:
        app.after(300, app.open_settings)
    app.mainloop()


if __name__ == "__main__":
    main()
