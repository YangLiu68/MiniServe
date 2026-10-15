"""``miniserve`` / ``python -m miniserve``: run the server, configured via MINISERVE_* env vars."""

from __future__ import annotations

import logging
import math
import sys

import uvicorn

from miniserve.config import Settings
from miniserve.logging_setup import configure_logging
from miniserve.server import create_app


class DrainingServer(uvicorn.Server):
    """On SIGTERM/SIGINT: fail readiness and refuse new work first, then let uvicorn stop
    accepting connections and wait (``timeout_graceful_shutdown``) for in-flight streams."""

    def __init__(self, config: uvicorn.Config, app) -> None:
        super().__init__(config)
        self._state = app.state.miniserve

    def handle_exit(self, sig, frame) -> None:
        self._state.begin_drain()
        super().handle_exit(sig, frame)


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    try:
        settings.check_bind_safety()
    except RuntimeError as e:
        print(f"miniserve: {e}", file=sys.stderr)
        sys.exit(2)
    if not settings.auth_enabled:
        logging.getLogger("miniserve").warning(
            "no API key configured: serving WITHOUT authentication", extra={"host": settings.host}
        )
    app = create_app(settings)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        log_config=None,  # keep our structured handler
        # 0 = cancel in-flight requests immediately (None would mean wait forever)
        timeout_graceful_shutdown=math.ceil(settings.drain_timeout_s),
    )
    DrainingServer(config, app).run()


if __name__ == "__main__":
    main()
