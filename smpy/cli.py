"""Command-line interface for SMPy."""

import logging
from argparse import ArgumentParser
from pathlib import Path

from smpy import run

def main(argv=None):
    """Run a mass mapping workflow from a YAML configuration file.

    Parameters
    ----------
    argv : `list` of `str`, optional
        Command-line arguments. If `None`, they are read from ``sys.argv``.
    """
    parser = ArgumentParser(prog="smpy", description="Create mass maps from a YAML configuration file.")
    parser.add_argument("-c", "--config", type=Path, required=True, metavar="PATH",
                        help="path to the YAML configuration file")
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"configuration file not found: {args.config}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.config)
