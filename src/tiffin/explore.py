from __future__ import annotations
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print(train.shape, later.shape, holdout.shape)
print(train.head())
print("Columns in train but not in holdout:", set(train.columns) - set(holdout.columns))
print(train.describe().round(1))