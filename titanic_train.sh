#!/usr/bin/env bash

set -eux

rm -rf output/titanic_model

python titanic_train.py