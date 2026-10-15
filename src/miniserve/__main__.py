"""``miniserve`` / ``python -m miniserve``: run the server, configured via MINISERVE_* env vars."""

from __future__ import annotations

import logging
import sys

import uvicorn

from miniserve.config import Settings
from miniserve.server import create_app


def main() -> None:
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings.check_bind_safety()
    except RuntimeError as e:
        print(f"miniserve: {e}", file=sys.stderr)
        sys.exit(2)
    if not settings.api_key:
        logging.getLogger("miniserve").warning(
            "MINISERVE_API_KEY is not set: serving WITHOUT authentication on %s", settings.host
        )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
