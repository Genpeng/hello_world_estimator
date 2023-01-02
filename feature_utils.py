# coding: utf-8

import tensorflow as tf


def _int64_feature(values):
    """
    Create a Int64List feature from a bool / enum / int32 / uint32 / int64 / uint64.

    Args:
        values: scalar or iterable, represents feature values

    Returns:
        Int64List feature
    """
    if isinstance(values, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def _float_feature(values):
    """
    Create a FloatList feature from a float / double.

    Args:
        values: scalar or iterable, represents feature values

    Returns:
        FloatList feature
    """
    if isinstance(values, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def _bytes_feature(values):
    """
    Create a BytesList feature from a bytes / string.

    Args:
        values: scalar or iterable, represents feature values

    Returns:
        BytesList feature
    """
    if isinstance(values, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
