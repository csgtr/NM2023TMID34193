#!/usr/bin/env python
# coding: utf-8

# In[20]:


#import necessary libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
#import imblearn
#from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score


# In[17]:



data=pd.read_csv("D:\\NMDS\\Churn_Modelling.csv")
data.head()


# In[ ]:





# In[21]:


data.info()


# In[24]:


#data.TotalCharges=pd.to_numeric(data.TotalCharges,error='coerce')
data.isnull().any()
                                


# In[25]:


data.isnull().sum()
                   


# In[26]:


data.describe()


# In[28]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.distplot(data['Tenure'])
plt.subplot(1,2,2)
sns.distplot(data['NumOfProducts'])


# In[38]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.distplot(data['CreditScore'])
plt.subplot(1,2,2)
sns.distplot(data['Balance'])


# In[39]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(data['Gender'])
plt.subplot(1,2,2)
sns.distplot(data['HasCrCard'])


# In[43]:


sns.barplot(x="CreditScore",y="EstimatedSalary",data=data)


# In[44]:


sns.heatmap(data.corr(),annot=True)


# In[45]:


sns.pairplot(data=data,markers=["^","v"],palette="Inferno")


# In[48]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_resamble,y_resamble,test_size=0.2,random_state=0)


# In[54]:


from sklearn.preprocessing import standarscaler
sc= standardscaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[56]:


#importing and building the Decision tree model
def logreg(x_train,x_test,y_train,y_test):
    lr=LogisticRegression(random_state=0)
    lr.fit(x_train,y_train)
    y_lr_tr=lr.predict(x_train)
    print(accuracy_score(y_lr_tr,y_))
    ypred_lr=lr.predict(x_test)
    print(accuracy_score(ypred_lr,y_test))
    print("***Logistic Regression***")
    print("confusion_Matrix")
    print(confusion_matrix(y_test,ypred_lr))
    print("classification Report")
    print(classification_report(y_test,ypred_lr))


# In[58]:


#printing the train accuracy and test accuracy respectively
logreg(x_train,x_test,y_train,y_test)


# In[ ]:


#importing and building the Decision tree model
def decisionTree(x_train,x_test,y_train,y_test):
    dtc=DecisionTreeClassifier(criterion="entropy",random_state=0)
    dec.fit(x_train,y_train)
    y_dt_tr=dtc.predict(x_train)
    print(accuracy_score(y_dt_tr,y_train))
    ypred_dt=dtc.predict(x_test)
    print(accuracy_score(ypred_dt,y_test))
    print("***Decision Tree***")
    print("confusion_matrix")
    print(confusion_matrix(y_test,ypred_dt))
    print("Classification Report")
    print(classification_report(y_test,ypred_dt))


# In[59]:


#printing the train accuracy and test accuracy respecftively
decisionTree(x_train,x_test,y_train,y_test)


# In[ ]:




