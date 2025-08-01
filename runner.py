"""Command line interface for SMPy mass mapping operations.

This module provides a simple command line tool to run SMPy mass mapping
operations using YAML configuration files. It serves as the primary entry
point usage within the repository.
"""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import smpy

def main(args):
    """Execute SMPy operation from command line arguments.

    Load configuration from the specified YAML file and execute the
    mass mapping operation using the parameters defined in the config.

    Parameters
    ----------
    args : `argparse.Namespace`
        Parsed command line arguments containing config file path.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file contains invalid parameters.

    Examples
    --------
    Run mass mapping with configuration file:

    >>> import argparse
    >>> args = argparse.Namespace(config=Path('my_config.yaml'))
    >>> main(args)
    """
    smpy.run(args.config)

if __name__ == "__main__":
    parser = ArgumentParser(description='Runner script for SMPy operations.')
    parser.add_argument(
        '-config', '-c',
        type=Path,
        help='Path to configuration file (.yaml)',
        required=True,
        metavar='PATH'
    )
    args = parser.parse_args()
    
    # Check if file exists
    if not args.config.is_file():
        raise ArgumentTypeError(f"Not a valid file: {args.config}")
    
    main(args)