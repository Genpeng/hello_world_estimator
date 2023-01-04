#!/usr/bin/env bash

set -eux

python split_titanic_dataset.py \
  --input_filepath "input/titanic/train.csv" \
  --output_dir "input/titanic"