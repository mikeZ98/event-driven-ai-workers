"""``edaw`` command line: ``edaw api`` serves the producer, ``edaw worker`` runs a worker.

Both read their configuration from ``EDAW_*`` environment variables.
"""

import argparse
import asyncio
import logging
from collections.abc import Sequence

import uvicorn

from event_driven_ai_workers import __version__
from event_driven_ai_workers.api.config import ApiSettings
from event_driven_ai_workers.worker_main import run_worker

APP_FACTORY = "event_driven_ai_workers.api.app:create_app"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edaw", description="Event-driven AI workers: producer API and arq worker."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("api", help="serve the FastAPI producer (EDAW_API_*)")
    worker = commands.add_parser(
        "worker", help="run an arq worker and its /metrics listener (EDAW_WORKER_*, EDAW_METRICS_*)"
    )
    worker.add_argument("--burst", action="store_true", help="exit once the queue is empty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "api":
        settings = ApiSettings()
        uvicorn.run(APP_FACTORY, factory=True, host=settings.host, port=settings.port)
    else:
        asyncio.run(run_worker(burst=args.burst))
    return 0
