#DataFrame.fillna #DataFrame.fillna  is used to fiil the nall data in the set for more data base
# imblearn over_sampling is been done in this version by random oversampling methods


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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


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

ros = RandomOverSampler(random_state=12)

traindata, testdata = train_test_split(data_1,stratify=data_1['loan_status'],test_size=.2)
testdata.reset_index(drop=True, inplace=True)
traindata.reset_index(drop=True, inplace=True)

sc=StandardScaler()
X_data_1 = traindata.drop('loan_status', axis=1)
y_data_1 = traindata['loan_status']
numerical = X_data_1.columns[(X_data_1.dtypes == 'float64') | (X_data_1.dtypes == 'int64')].tolist()
X_data_1[numerical] = sc.fit_transform(X_data_1[numerical])

Xunb = traindata.drop('loan_status', axis=1)
yunb = traindata['loan_status']
numerical = Xunb.columns[(Xunb.dtypes == 'float64') | (Xunb.dtypes == 'int64')].tolist()
Xunb[numerical] = sc.fit_transform(Xunb[numerical])

Xunb, yunb= ros.fit_resample(Xunb, yunb)


#print(yunb.shape)

### Declaring x and y for test data
Xte = testdata.drop('loan_status', axis=1)
yte = testdata['loan_status']
numerical = Xte.columns[(Xte.dtypes == 'float64') | (Xte.dtypes == 'int64')].tolist()
Xte[numerical] = sc.fit_transform(Xte[numerical])
print(data_1)

RFC=RandomForestClassifier(n_estimators=15, random_state=12)
RFC.fit(Xunb,yunb)
predict=RFC.predict(Xte)

plt.figure(figsize=(20,20))
cm = confusion_matrix(yte, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

ax = sns.heatmap(cm, annot=True, cmap='Greens')
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.axis('equal')

score= cross_val_score(RFC,X_data_1,y_data_1,cv=10)
print("********************************")
print("Score=",score)
print("*********************************")

plt.title("Distribution of Loan Status")
plt.show()
