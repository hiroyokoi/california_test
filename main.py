import numpy as np
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

import streamlit

print(np.__version__)

#--------------model --------------#
model = pickle.load(open('utils/model.pkl', 'rb'))
ss = pickle.load(open('utils/ss_scaler.pkl', 'rb'))

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

X_test_trans = ss.transform(X_test)

model.fit(X_test_trans, y_test)
pred = model.predict(X_test_trans)
print(np.sqrt(mean_squared_error(y_test, pred)))
print(r2_score(y_test, pred))