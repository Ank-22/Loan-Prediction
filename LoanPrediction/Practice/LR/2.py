#This program is for linear Regrassion using sklearn, pandas, numpy

import pandas as pd
import numpy as np 

df_train=pd.read_csv('C:/Users/ankitarbins/Documents/PythonPrograming/LoanPrediction/Practice/LR/train.csv')
df_test=pd.read_csv('C:/Users/ankitarbins/Documents/PythonPrograming/LoanPrediction/Practice/LR/test.csv')

x_train=df_train['x']
y_train=df_train['y']
x_test=df_test['x']
y_test=df_test['y']

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(y_pred.data.shape)
print(r2_score(y_test,y_pred))
#output : R2= 0.8104548890346173