#!/usr/bin/env bash

set -eux

python split_titanic_dataset.py \
  --input_filepath "input/titanic/train.csv" \
  --output_dir "input/titanic"

python prepare_titanic_tfrecord.py \
  --input_filepath "input/titanic/train_sampled.csv" \
  --output_filepath "input/titanic/train/train.tfrecords" \
  --compress \
  --chunk_size 1000 \
  --log_steps 100 \
  --training

python prepare_titanic_tfrecord.py \
  --input_filepath "input/titanic/eval_sampled.csv" \
  --output_filepath "input/titanic/eval/eval.tfrecords" \
  --compress \
  --chunk_size 1000 \
  --log_steps 100 \
  --training

python prepare_titanic_tfrecord.py \
  --input_filepath "input/titanic/test.csv" \
  --output_filepath "input/titanic/test/test.tfrecords" \
  --compress \
  --chunk_size 1000 \
  --log_steps 100