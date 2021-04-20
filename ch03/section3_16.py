import pandas as pd
import numpy as np
import torch
import torch.nn as nn

train_data = pd.read_csv(r"../Datasets/HousePricesAdvancedRegressionTeniques/train.csv")
test_data = pd.read_csv(r"../Datasets/HousePricesAdvancedRegressionTeniques/test.csv")
print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])