# coding: utf-8

"""
Split Titanic train.csv into a training set and a validation set.

Author: Genpeng Xu (xgp1227atgmail.com)
"""

import argparse
import logging
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_filepath",
        type=str,
        default=None,
        required=True,
        help="The file path of input data to split into training and validation set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory to store result.",
    )
    return parser.parse_args()


def print_arguments(args):
    print("=============== Configuration Arguments ===============")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("=======================================================")


def run(args):
    t0 = time.time()
    logger.info("start split titanic data...")

    df = pd.read_csv(args.input_filepath)
    train_df, eval_df = train_test_split(
        df, test_size=0.2, shuffle=True, stratify=df["Survived"], random_state=2023
    )
    train_df.to_csv(os.path.join(args.output_dir, "train_sampled.csv"))
    eval_df.to_csv(os.path.join(args.output_dir, "eval_sampled.csv"))

    t1 = time.time()
    logger.info(f"finish split titanic data, time: {t1 - t0}")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    run(args)
