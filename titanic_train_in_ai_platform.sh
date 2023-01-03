#!/usr/bin/env bash

set -eux

ROOT_DIR=/home/mlp/notebooks/gerry.xu/hello_world_estimator

export PYTHONPATH=$ROOT_DIR
export HADOOP_USER_NAME=hdfs

python -u $ROOT_DIR/titanic_train.py 2>&1 | tee /apps/logs/test.log