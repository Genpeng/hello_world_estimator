# coding: utf-8

"""
Split Titanic train.csv into a training set and a validation set.

Author: Genpeng Xu (xgp1227atgmail.com)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/titanic/train.csv")
train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=True, stratify=df["Survived"], random_state=2023)
train_df.to_csv("datasets/titanic/train_sampled.csv")
eval_df.to_csv("datasets/titanic/eval_sampled.csv")
