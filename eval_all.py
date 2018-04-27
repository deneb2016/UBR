
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate multiple models')

    parser.add_argument('--dir', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for model_name in os.listdir(args.dir):
        if model_name.endswith('.pth'):
            os.system('python eval.py --model_path %s' % os.path.join(args.dir, model_name))

