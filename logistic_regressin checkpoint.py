#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
data=pd.read_csv("logistic_regression_dataset.csv")
data


# In[19]:


data.isnull().sum()


# In[20]:


data['education'].fillna(data['education'].mean(),inplace=True)
data


# In[21]:


data.isnull().sum()


# In[32]:


data['cigsPerDay'].fillna(data['cigsPerDay'].mean(),inplace=True)


# In[23]:


data['BPMeds'].value_counts()


# In[24]:


data['BPMeds']=data['BPMeds'].fillna(0)
data


# In[25]:


data.isnull().sum()


# In[26]:


data['TenYearCHD'].value_counts()


# In[27]:


data=data.dropna(subset=['TenYearCHD'])
data


# In[28]:


data.isnull().sum()


# In[29]:


data['totChol'].value_counts()


# In[31]:


data['totChol'].fillna(data['totChol'].median(),inplace=True)
data


# In[33]:


data.isnull().sum()


# In[34]:


data['BMI'].value_counts()


# In[42]:


data['BMI'].fillna(data['BMI'].median(),inplace=True)
data


# In[43]:


data.isnull().sum()


# In[44]:


data['heartRate'].value_counts()


# In[45]:


data['heartRate']=data['heartRate'].fillna(75.0)


# In[46]:


data.isnull().sum()


# In[47]:


data['glucose'].value_counts()


# In[48]:


data=data.dropna(subset=['glucose'])


# In[49]:


data


# In[50]:


data.isnull().sum()


# In[52]:


data["TenYearCHD"]=data["TenYearCHD"].map({1.0: 1, 0.0: 0})  


# In[53]:


data


# In[56]:


data["is_smoking"]=data["is_smoking"].map({"YES": 1, "NO": 0})
data


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
x = data[['is_smoking','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]
y = data['TenYearCHD']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)  #splitting data with test size of 25%

logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("Accuracy={:.3f}".format(logreg.score(x_test, y_test)))





# In[71]:


import seaborn as sns
sns.regplot(x='age',y='TenYearCHD',data=data)


# In[72]:


# matrix confusion
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)


# In[73]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[78]:


#ROC / AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred , pos_label=1)


# auc scores
auc_score1 = roc_auc_score(y_test, y_pred)
auc_score2 = roc_auc_score(y_test, y_pred)

print(auc_score1, auc_score2)


# In[79]:


import matplotlib.pyplot as plt
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

