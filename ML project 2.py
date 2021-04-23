#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:11:44 2020

@author: bhaskaryuvaraj
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
by=pd.read_csv('/Users/bhaskaryuvaraj/Downloads/Banking.csv')
by_trial=pd.read_csv('/Users/bhaskaryuvaraj/Downloads/Banking.csv')

len(by)
len(by.columns)
#no. of rows and columns are 41188 and 21

by.describe()
by.columns
by.dtypes
by.head()
#-------------------------------EDA------------------------------------



by_yes=by[by['y']==1]
by_no=by[by['y']==0]


by_yes.groupby('age')['y'].count().plot(kind='line',color='green')
#from the graph, it is clear that people around the age of 20-40 had taken the loan

by_no.groupby('age')['y'].count().plot(kind='line',color='green')
#from the graph, it is clear that mostly people around the age of 30-50 had taken the loan. Hence both the graphs 
#doesnt give any clear picture about the age relation

by_yes.groupby('marital')['y'].count().plot(kind='line')
#most of the married people had gone for it

by_no.groupby('marital')['y'].count().plot(kind='line')
#again most of married people had not gone for it. lets check the next factor

by['education'].unique()
by_yes.groupby('education')['y'].count().plot(kind='bar')
#people with education in university degree moatly had gone for it

by_no.groupby('education')['y'].count().plot(kind='bar')
#the data is similar with this factor too

by_yes.groupby('default')['y'].count().plot(kind='bar')
by_no.groupby('default')['y'].count().plot(kind='bar')
#most number with no default had gone for it as well as not gone for it so not helpful

by_yes.groupby('housing')['y'].count().plot(kind='bar')
by_no.groupby('housing')['y'].count().plot(kind='bar')
#people who have houses have accepted the most as well as rejected

by_yes.groupby('loan')['y'].count().plot(kind='bar')
by_no.groupby('loan')['y'].count().plot(kind='bar')
#people with no loan had accepted and rejcted the most
 
by_yes.groupby('contact')['y'].count().plot(kind='bar')
by_no.groupby('contact')['y'].count().plot(kind='bar')
 #cellular mode is more effective compared to other modes

by_yes.groupby('month')['y'].count().plot(kind='bar')
by_yes.groupby(['month','day_of_week'])['y'].count().plot(kind='bar')

by_no.groupby('month')['y'].count().plot(kind='bar')
by_no.groupby('day_of_week')['y'].count().plot(kind='bar')
#in the month of may most of the people responded 
by_yes.groupby('duration')['y'].count().plot(kind='line')
by.plot(kind='scatter',x='y',y='duration')
by_no.groupby('duration')['y'].count().plot(kind='line')
#more number of people who had spoken in phone around 50-1000 seconds have accepted and most people 
#who have spoken less than 500 sec have rejected

by_yes.groupby('campaign')['y'].count().plot(kind='bar')
by_no.groupby('campaign')['y'].count().plot(kind='bar')
#again both graphs for accepted and rejected had most no. of people with 1 no. of campaign

by2=by[by['job']=='unknown']
by2.groupby(['job','education'])['education'].count().plot(kind='bar')

#-------------------------------------method2----------------------------------------------


by_trial=pd.read_csv('/Users/bhaskaryuvaraj/Downloads/Banking.csv')
by_trial['y'].replace({0:'No',1:'Yes'},inplace=True)

by_trial.groupby(['age','y'])['y'].count().plot(kind='line')
#refer above method for results

by_trial.groupby(['marital','y'])['y'].count().plot(kind='bar')
#from the graph it is clear that most of the married people rejected and accepted as well

by_trial.groupby(['education','y'])['y'].count().plot(kind='bar')
#most of the university degree holders said no and yes

by_trial.groupby(['default','y'])['y'].count().plot(kind='bar')
#mostly with default no have accepted and rejected

by_trial.groupby(['housing','y'])['y'].count().plot(kind='bar')
#same result as above method

by_trial.groupby(['loan','y'])['y'].count().plot(kind='bar')
#most of the people with no loan have rejected and accepted

by_trial.groupby(['contact','y'])['y'].count().plot(kind='bar')
#mostly cellular accepted and rejected

by_trial.groupby(['month','y'])['y'].count().plot(kind='line')
#most of the people in the month of may had accepted and rejected as well


#-------------------------------------end of EDA-------------------------------------------

#to find null values
by.isnull().sum()
by.columns.unique()
by['education'].unique()
by.replace({'unknown':np.nan},inplace=True)
by.dropna(inplace=True)
by['previous'].unique()
by['campaign'].unique()

#to check for outliers
plt.boxplot(by['duration'])
plt.boxplot(by['campaign']) #outlier treatement is not necessary
plt.boxplot(by['pdays'])      #if you do outlier treatement everything is deleted
plt.boxplot(by['previous'])  #outlier treatement affects the data as above
plt.boxplot(by['emp_var_rate'])
plt.boxplot(by['cons_price_idx'])
plt.boxplot(by['cons_conf_idx'])
plt.boxplot(by['euribor3m'])
plt.boxplot(by['nr_employed'])
plt.boxplot(by['age'])

#outlier treatement

def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

by=remove_outlier(by,'duration')
plt.boxplot(by['duration'])
by=remove_outlier(by,'cons_conf_idx')
plt.boxplot(by['cons_conf_idx'])
by=remove_outlier(by,'age')
plt.boxplot(by['age'])

#checking the unknown catogory relation with education
by2=by[by['job']=='unknown']
by2.groupby(['job','education'])['education'].count().plot(kind='bar')
#most of the unknown catory job is under unknown catory section in education. Hence unknown catory is put 
#under others section.
by['job'].replace(['technician','services','housemaid'],['blue-collar','blue-collar','blue-collar',],
  inplace=True)
by['job'].replace(['management','retired','entrepreneur','admin.','self-employed'],
    ['white-collar','white-collar','white-collar','white-collar','white-collar',],inplace=True)
by['job'].replace(['unemployed','unknown','student'],['others','others','others'],inplace=True)
by['job'].unique()
by['default'].replace(['unknown','yes','no'],['default_unknown','default_yes','default_no'],inplace=True)
by['housing'].unique()
by['housing'].replace(['unknown','yes','no'],['housing_unknown','housing_yes','housing_no'],inplace=True)
by['loan'].unique()
by['loan'].replace(['unknown','yes','no'],['loan_unknown','loan_yes','loan_no'],inplace=True)
by['marital'].replace('unknown','marital_unknown',inplace=True)
by1=by[by['education']=='unknown']
by1.groupby(['education','job'])['job'].count().plot(kind='bar')
#from the above graph it is clear that most of the unknown people education work under blue-collar.
#hence their education  comes under basic catogory
by['education'].unique()
#grouping the education catogory
by['education'].replace(['basic.4y','unknown','basic.9y','basic.6y','illiterate'],
  ['basic','basic','basic','basic','basic'],inplace=True)
by['education'].replace(['university.degree','professional.course'],['degree','degree'],inplace=True)


#creating the dummy columns and eliminating unwanted columns
by.columns.unique()
dummy1=pd.get_dummies(by['job'])
dummy2=pd.get_dummies(by['marital'])
dummy3=pd.get_dummies(by['education'])
dummy4=pd.get_dummies(by['default'])
dummy5=pd.get_dummies(by['housing'])
dummy6=pd.get_dummies(by['loan'])
dummy7=pd.get_dummies(by['contact'])
dummy8=pd.get_dummies(by['month'])
dummy9=pd.get_dummies(by['day_of_week'])
dummy10=pd.get_dummies(by['poutcome'])

#now creating the master data
ab=pd.concat([by,dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7,dummy8,dummy9,dummy10],axis=1)
master_data=ab.drop(ab.columns[[1,2,3,4,5,6,7,8,9,14]],axis=1)
#---------------------Feature Selection-------------------------------------------------
#they provide the same information. Hence we will remove them
correlated_features = set()
correlation_matrix = master_data.drop('y', axis=1).corr()

for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)
#Check correlated features            
print(correlated_features)
#seperating the dependent and independent variables
y=master_data['y'].copy()
x=master_data.drop('y', axis=1)
x=x.drop(correlated_features,axis=1)

#create training and test data by splitting x and y into 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)
x_train.columns

#-------------------------------logistic regression starts--------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

print(logreg.score(x_train,y_train))
#accuracy=0.9462211043238711
pred_y=logreg.predict(x_test)

#accuracy of test model
logreg.score(x_test,y_test)
# accuracy=0.9429902189101071

#create confusion matrix
from sklearn.metrics import confusion_matrix
c_m=confusion_matrix(y_test,pred_y)
# accuracy=(9857+266)/10735=0.9429902189101071
#-------------------------------logistic regression ends--------------------------------------

#---------------------------------KNN starts---------------------------------------------------
#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(x_train, y_train)

Knn_pred = knn.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#Print accuracy score
knn.score(x_test,y_test)
#accuracy= 0.9402887750349325
#--------------------------------KNN ends----------------------------------------------------

#---------------------------------Decision tree starts---------------------------------------------------
#classification with scikit-learn decision tree
from sklearn import tree
d_tree = tree.DecisionTreeClassifier()
d_tree.fit(x_train, y_train)
         
dtree_pred=d_tree.predict(x_test)
d_tree.score(x_test,y_test)
#accuracy=0.928365160689334
#---------------------------------Decision tree ends---------------------------------------------------

#---------------------------------Random forest starts---------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV#(estimated cross validation)
from sklearn.model_selection import StratifiedKFold
r_forest = RandomForestClassifier(random_state=101)
r_forest_ecv = RFECV(estimator=r_forest, step=1, cv=StratifiedKFold(10), scoring='accuracy')

#Fitting the model
r_forest_ecv.fit(x_train, y_train)
rf_pred=r_forest_ecv.predict(x_test)
r_forest_ecv.score(x_test,y_test)
#accuracy=0.9488588728458314
#---------------------------------Random forest ends---------------------------------------------------




