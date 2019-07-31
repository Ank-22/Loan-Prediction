# This prgram is for linear regrasssion using opreations
import numpy as np
import pandas as pd

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
n=700
alpha=0.0001

a_0=np.zeros((n,))
a_1=np.zeros((n,))

x_train=np.resize(x_train,(700,))
y_train=np.resize(y_train,(700,))
epochs=0

while(epochs<1000):
    y = a_0 + a_1 * x_train
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    a_0 = a_0 - alpha * 2 * np.sum(error)/n 
    a_1 = a_1 - alpha * 2 * np.sum(error * x_train)/n
    epochs += 1
    if(epochs%10 == 0):
        print(mean_sq_er)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
a_0=np.resize(a_0,(300,))
a_1=np.resize(a_1,(300,))
y_prediction = a_0 + a_1 * x_test
y_prediction=np.resize(y_prediction,(300,))
print('R2 score : ',r2_score(y_test,y_prediction))

y_plot = []
for i in range(100):
    y_plot.append(a_0 + a_1 * i)
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='green',label='GT')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
plt.legend()
plt.show()


#output : R2 score :  0.8760236288881302
