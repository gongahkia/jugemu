from __future__ import annotations

import sys


def _run(subcommand: str) -> None:
    from .cli import app

    argv0 = sys.argv[0]
    sys.argv = [argv0, subcommand, *sys.argv[1:]]
    app()


def ingest_main() -> None:
    _run("ingest")


def train_main() -> None:
    _run("train")


def chat_main() -> None:
    _run("chat")
