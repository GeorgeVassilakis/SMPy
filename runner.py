from smpy import run
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

def main(args):
    run.run(args.config)

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