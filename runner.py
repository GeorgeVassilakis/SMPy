from SMPy import run
from argparse import ArgumentParser
import os

def main(args):
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config {config_path} not found")
    else:
        run.run(config_path)

if __name__ == "__main__":
    parser = ArgumentParser(description='Runner script for SMPy operations.')
    parser.add_argument('-config', '-c', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    main(args)