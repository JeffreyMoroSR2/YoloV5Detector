import sys
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='external/carplates_det.pt', help='Path to config.json')
    parser.add_argument('--debug', type=bool, default=False, help='Enable logging')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parse_opt()
