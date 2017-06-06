import pandas as pd
import numpy as np
from sklearn import preprocessing

# load csv with pandas
df = pd.read_csv('../data/covtype.data')
data = df.values

# split data and label
X, y = np.hsplit(data, np.array([54]))

# extract previous 10 dimensions and scale them to unit variable
X_scaled = preprocessing.scale(X[:, 0:10])

