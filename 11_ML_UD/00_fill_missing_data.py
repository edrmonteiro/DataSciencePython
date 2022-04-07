# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

baseDeDados = pd.read_csv('train_missing.csv')
X = baseDeDados.iloc[:,:].values
print(X)

from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values=np.nan, strategy='median')
Imputer = Imputer.fit(X[:,5:6])
X = Imputer.transform(X[:,5:6]).astype(str)
X = np.insert(X, 0, baseDeDados.iloc[:,0].values, axis=1)

print(X)
