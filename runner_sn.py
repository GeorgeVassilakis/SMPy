from SMPy.SNL import run
from argparse import ArgumentParser
import os
import argparse


def main(args):
        
    config_path = args.config
    if os.path.isfile(config_path) != True:
        raise FileNotFoundError(f"config {config_path} not found")
    
    else:
        run.run(config_path)


if __name__ == "__main__":

    # Define config arguments
    parser = argparse.ArgumentParser(
        description='Runner script for SMPy operations.'
    )

    parser.add_argument(
        '-config', '-c', type=str, help='Configuration file', required=True
    )

    # Parse arguments
    args = parser.parse_args()

    # Go!
    main(args)
