"""Run a staged policy server with host extensions preloaded for local audit."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_local_fast_server.py POLICY_SERVER.py [args ...]")
    server_path = Path(sys.argv[1]).resolve()
    if not server_path.is_file():
        raise FileNotFoundError(server_path)
    # The staged vendor tree targets the official torch 2.11 image.  The
    # current audit venv has a matching host extension build; preload it
    # before policy_server.py inserts its vendor directory.
    import torchvision  # noqa: F401
    import sentencepiece  # noqa: F401

    sys.argv = [str(server_path), *sys.argv[2:]]
    runpy.run_path(str(server_path), run_name="__main__")


if __name__ == "__main__":
    main()
