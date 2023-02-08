import cv2
import argparse
from threading import Thread
from modules.worker import Worker


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='external/weights/detector/carplates_det.pt',
                        help='Path to config.json')
    parser.add_argument('--debug', type=bool, default=False, help='Enable logging')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cmd_opt = parse_opt()
    worker = Worker(cmd_opt.config, cmd_opt.debug).start()
    worker.run()
