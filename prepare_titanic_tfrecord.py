# coding: utf-8

"""
Convert Titanic Datasets into TFRecord files.

Author: Genpeng Xu
"""

import argparse
import logging
import time

import pandas as pd
import tensorflow as tf

from feature_utils import _int64_feature, _float_feature, _bytes_feature

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def get_compression_type():
    if tf.__version__ >= "2.0":
        return tf.compat.v1.io.TFRecordCompressionType.GZIP
    else:
        return tf.io.TFRecordCompressionType.GZIP


def create_titanic_example_from_series(row):
    feature_map = {
        "PassengerId": _int64_feature(int(row.PassengerId)),
        "Survived": _int64_feature(int(row.Survived)),
        "Pclass": _int64_feature(int(row.Pclass)),
        "Name": _bytes_feature(str(row.Name).encode("utf-8")),
        "Sex": _bytes_feature(str(row.Sex).encode("utf-8")),
        "Age": _float_feature(float(row.Age)),
        "SibSp": _int64_feature(int(row.SibSp)),
        "Parch": _int64_feature(int(row.Parch)),
        "Ticket": _bytes_feature(str(row.Ticket).encode("utf-8")),
        "Fare": _float_feature(float(row.Fare)),
        "Cabin": _bytes_feature(str(row.Cabin).encode("utf-8")),
        "Embarked": _bytes_feature(str(row.Embarked).encode("utf-8")),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_map))


def convert_titanic_to_tfrecord(
    input_filepath: str,
    output_filepath: str,
    compress: bool = False,
    chunk_size: int = 10000,
    log_steps: int = 100,
    training: bool = True,
) -> None:
    options = None
    if compress:
        options = tf.io.TFRecordOptions(compression_type=get_compression_type())
    with tf.io.TFRecordWriter(output_filepath, options=options) as writer:
        with pd.read_csv(input_filepath, iterator=True) as reader:
            while True:
                try:
                    chunk = reader.get_chunk(chunk_size)

                    # fill missing value
                    chunk["Age"] = chunk["Age"].fillna(-1.0)
                    chunk["Cabin"] = chunk["Cabin"].fillna("-1")
                    chunk["Embarked"] = chunk["Embarked"].fillna("-1")

                    if not training:
                        chunk["Survived"] = 0

                    for i, row in chunk.iterrows():
                        try:
                            example = create_titanic_example_from_series(row)
                        except Exception as e:
                            logger.info(
                                f"convert to example error, msg: {e}, row: {row.to_dict()}"
                            )
                            continue

                        if (i + 1) % log_steps == 0:
                            logger.info(f"example {i + 1}:\n{example}")

                        writer.write(example.SerializeToString())

                except StopIteration:
                    break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_filepath",
        type=str,
        default=None,
        required=True,
        help="The file of input data to convert to tfrecord file.",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default=None,
        required=True,
        help="The tfrecord file stores processed data.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="The output file format is GZIP or not.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        required=False,
        help="Number of rows of data read into at one time.",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        required=False,
        help="Number of update steps between two logs.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        default=False,
        help="The output file format is GZIP or not.",
    )
    return parser.parse_args()


def print_arguments(args):
    print("=============== Configuration Arguments ===============")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("=======================================================")


def run(args):
    t0 = time.time()
    logger.info("start prepare tfrecord data...")

    convert_titanic_to_tfrecord(
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        compress=args.compress,
        chunk_size=args.chunk_size,
        log_steps=args.log_steps,
        training=args.training,
    )

    t1 = time.time()
    logger.info(f"finish prepare tfrecord data, time: {t1 - t0}")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    run(args)
