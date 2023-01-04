#!/usr/bin/env bash

set -eux

python prepare_titanic_tfrecord.py \
  --input_filepath "input/titanic/train_sampled.csv" \
  --output_filepath "input/titanic/train/train.tfrecords" \
  --chunk_size 1000 \
  --log_steps 200 \
  --training

python prepare_titanic_tfrecord.py \
  --input_filepath "input/titanic/eval_sampled.csv" \
  --output_filepath "input/titanic/eval/eval.tfrecords" \
  --chunk_size 1000 \
  --log_steps 200 \
  --training

python prepare_titanic_tfrecord.py \
  --input_filepath "input/titanic/test.csv" \
  --output_filepath "input/titanic/test/test.tfrecords" \
  --chunk_size 1000 \
  --log_steps 200