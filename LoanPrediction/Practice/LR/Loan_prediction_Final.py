### Loan Prediction By Ankit Sinha : 14/06/2019
### Aim : To use the previous data for both test and train for prediction that a preson will be elligable for loan in future or not.

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


data_1= pd.read_csv('C:/Users/ankitarbins/Documents/PythonPrograming/LoanPrediction/Dataforloan/loan.csv', error_bad_lines=True)
###Taking the Required Data from the Files
data_1 = data_1.filter(['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership',
                    'annual_inc','verification_status','purpose','dti','delinq_2yrs','loan_status'])
### Removing the null data from the row.
data_1=data_1.dropna(axis=0)
### removing more data for precision
data_1.drop(['installment','grade','sub_grade','verification_status','term']
           , axis=1, inplace = True)
### manuplatting the data for easy Learing 
data_1= data_1[data_1.loan_status != 'Current']
data_1= data_1[data_1.loan_status != 'Late (16-30 days)']
data_1= data_1[data_1.loan_status != 'In Grace Period']
data_1= data_1[data_1.loan_status != 'Does not meet the credit policy. Status:Fully Paid']
data_1= data_1[data_1.loan_status != 'Late (31-120 days)']
data_1['loan_status'] = data_1['loan_status'].replace({'Charged Off':'Default'})
data_1['loan_status'] = data_1['loan_status'].replace({'Does not meet the credit policy. Status:Charged Off':'Default'})      


numerical = data_1.columns[data_1.dtypes == 'float64']
for i in numerical:
    if data_1[i].min() > 0:
        transformed, lamb = boxcox(data_1.loc[data_1[i].notnull(), i])
        if np.abs(1 - lamb) > 0.02:
            data_1.loc[data_1[i].notnull(), i] = transformed

### Spliting the data in 2 for trains and testing


data_1 = pd.get_dummies(data_1, drop_first=True)


traindata, testdata = train_test_split(data_1,test_size=.4,random_state=1939)
testdata.reset_index(drop=True, inplace=True)
traindata.reset_index(drop=True, inplace=True)

sc=StandardScaler()

Xunb = traindata.drop('loan_status', axis=1)
yunb = traindata['loan_status']
numerical = Xunb.columns[(Xunb.dtypes == 'float64') | (Xunb.dtypes == 'int64')].tolist()
Xunb[numerical] = sc.fit_transform(Xunb[numerical])

#print(yunb.shape)

### Declaring x and y for test data
Xte = testdata.drop('loan_status', axis=1)
yte = testdata['loan_status']
numerical = Xte.columns[(Xte.dtypes == 'float64') | (Xte.dtypes == 'int64')].tolist()
Xte[numerical] = sc.fit_transform(Xte[numerical])
print(data_1)

LGR=LogisticRegression(C=.0005)
LGR.fit(Xunb,yunb)
predict=LGR.predict(Xte)

plt.figure(figsize=(20,20))
cm = confusion_matrix(yte, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

ax = sns.heatmap(cm, annot=True, cmap='Greens')
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.axis('equal')

plt.title("Distribution of Loan Status")
plt.show()
#cd Documents\PythonPrograming\LoanPrediction\Practice\LR\