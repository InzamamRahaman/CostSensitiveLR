# File size of dataset is too large for GitHub.
# Download the csv file from the link provided: https://www.kaggle.com/mlg-ulb/creditcardfraud.
# This script can then be executed to convert the csv file to the npy file.

import numpy as np
import pandas as pd

df = pd.read_csv('creditcard.csv')
print(df.columns)
v = df.to_numpy(copy=True)
np.save('credit_card.npy', v)

