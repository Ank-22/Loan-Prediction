#DataFrame.fillna #DataFrame.fillna  is used to fiil the nall data in the set for more data base
# imblearn over_sampling is been done in this version by random oversampling methods
#CV is been used here !


#cd Documents\PythonPrograming\LoanPrediction\Practice\LR\

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:44:48 2019

@author: ankitarbins
"""
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
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold


data_1= pd.read_csv('C:/Users/ankitarbins/Documents/PythonPrograming/LoanPrediction/Dataforloan/loan.csv', error_bad_lines=True)

###Taking the Required Data from the Files
data_1 = data_1.filter(['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership',
                    'annual_inc','verification_status','purpose','dti','delinq_2yrs','loan_status'])

#data_null = pd.DataFrame({'Count': data_1.isnull().sum(), 'Percent': 100*data_1.isnull().sum()/len(data_1)})
###printing columns with null count more than 0
#print(data_null[data_null['Count'] > 0]) 

### Removing the null data from the row.
data_1=data_1.fillna(method='ffill')

data_null = pd.DataFrame({'Count': data_1.isnull().sum(), 'Percent': 100*data_1.isnull().sum()/len(data_1)})
###printing columns with null count more than 0
print(data_null[data_null['Count'] > 0]) 
print(data_1)



'''plt.figure(figsize=(20,20))
sns.set_context("paper", font_scale=1)

##finding the correllation matrix and changing the categorical data to category for the plot.
sns.heatmap(data_1.assign(grade=data_1.grade.astype('category').cat.codes,
                         sub_g=data_1.sub_grade.astype('category').cat.codes,
                         term=data_1.term.astype('category').cat.codes,
                        emp_l=data_1.emp_length.astype('category').cat.codes,
                         ver =data_1.verification_status.astype('category').cat.codes,
                        home=data_1.home_ownership.astype('category').cat.codes,
                        purp=data_1.purpose.astype('category').cat.codes).corr(), 
                         annot=True, cmap='bwr',vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.show()'''


#print(data_1['loan_status'].unique())

'''m=data_1['loan_status'].value_counts()
m=m.to_frame()
m.reset_index(inplace=True)
m.columns=['loan_status','count']
plt.subplots(figsize=(20,8))
sns.barplot(y='count', x='loan_status', data=m)
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Loan Status in our Dataset")
plt.show()'''

### manuplatting the data for easy Learing 
data_1= data_1[data_1.loan_status != 'Current']
data_1= data_1[data_1.loan_status != 'Late (16-30 days)']
#data_1= data_1[data_1.loan_status != 'In Grace Period']
data_1= data_1[data_1.loan_status != 'Does not meet the credit policy. Status:Fully Paid']
data_1= data_1[data_1.loan_status !='Does not meet the credit policy. Status:Charged Off']
data_1= data_1[data_1.loan_status != 'Late (31-120 days)']
data_1 = data_1[data_1.loan_status != 'Issued']
data_1['loan_status'] = data_1['loan_status'].replace({'Charged Off':'Default'})
data_1['loan_status'] = data_1['loan_status'].replace({'In Grace Period':'Default'})
data_1.loan_status=data_1.loan_status.astype('category').cat.codes
data_1.delinq_2yrs=data_1.delinq_2yrs.astype('category').cat.codes



#print(data_1['loan_status'].unique())
#print(data_1['loan_status'].value_counts())

numerical = data_1.columns[data_1.dtypes == 'float64']
for i in numerical:
    if data_1[i].min() > 0:
        transformed, lamb = boxcox(data_1.loc[data_1[i].notnull(), i])
        if np.abs(1 - lamb) > 0.02:
            data_1.loc[data_1[i].notnull(), i] = transformed

### Spliting the data in 2 for trains and testing


data_1 = pd.get_dummies(data_1, drop_first=True)
ros = RandomOverSampler(random_state=1)

## Changes Are been done here 
sc=StandardScaler()
skf = StratifiedKFold(n_splits=10, shuffle= True)
X = data_1.drop('loan_status', axis=1)
y = data_1['loan_status']
numerical = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')].tolist()
X[numerical] = sc.fit_transform(X[numerical]) 
X, y= ros.fit_resample(X, y)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]








#print(yunb.shape)

### Declaring x and y for test data

LGR=LogisticRegression(C=1)
LGR.fit(X_train,y_train)
predict=LGR.predict(X_test)

plt.figure(figsize=(20,20))
cm = confusion_matrix(y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

ax = sns.heatmap(cm, annot=True, cmap='Greens')
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.axis('equal')

plt.title("Distribution of Loan Status")
plt.show()
