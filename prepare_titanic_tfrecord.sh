#!/usr/bin/env bash

set -eux

python split_titanic_dataset.py

python prepare_titanic_tfrecord.py \
  --input_filepath "datasets/titanic/train_sampled.csv" \
  --output_filepath "datasets/titanic/train/train.tfrecords" \
  --compress \
  --chunk_size 1000 \
  --log_steps 100 \
  --training

python prepare_titanic_tfrecord.py \
  --input_filepath "datasets/titanic/eval_sampled.csv" \
  --output_filepath "datasets/titanic/eval/eval.tfrecords" \
  --compress \
  --chunk_size 1000 \
  --log_steps 100 \
  --training

python prepare_titanic_tfrecord.py \
  --input_filepath "datasets/titanic/test.csv" \
  --output_filepath "datasets/titanic/test/test.tfrecords" \
  --compress \
  --chunk_size 1000 \
  --log_steps 100