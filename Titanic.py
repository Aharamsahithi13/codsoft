#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# In[4]:


df = pd.read_csv(r'C:\Users\sahithi aharam\Downloads\titanic\tested.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().any()


# In[8]:


df.isnull().sum()


# In[143]:


df["Age"].fillna(df["Age"].mean(),inplace=True)


# In[145]:


df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)


# In[146]:


df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[147]:


df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[148]:


df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[149]:


df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[150]:


g = sn.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[151]:


grid = sn.FacetGrid(df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[152]:


grid = sn.FacetGrid(df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sn.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[153]:


grid = sn.FacetGrid(df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sn.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[154]:


df = df.drop(['Ticket', 'Cabin'], axis=1)


# In[155]:


df


# In[156]:


x = df.drop('Survived',axis=1)
y = df['Survived']


# In[157]:


x.head()


# In[158]:


type(x)


# In[159]:


type(y)


# In[160]:


le = LabelEncoder()
x.PassengerId = le.fit_transform(x.PassengerId)
x.head()


# In[161]:


y = le.fit_transform(y)
y


# In[162]:


x.Pclass = le.fit_transform(x.Pclass)
x.head()


# In[163]:


x.Sex = le.fit_transform(x.Sex)
x.head()


# In[164]:


x.Age = le.fit_transform(x.Age)
x.head()


# In[165]:


x.Fare = le.fit_transform(x.Fare)
x.head()


# In[166]:


x.Embarked = le.fit_transform(x.Embarked)
x.head()


# In[167]:


ms = MinMaxScaler()


# In[168]:


x_scaled= pd.DataFrame(ms.fit_transform(x),columns=x.columns)
x_scaled.head()


# In[169]:


x.info()


# In[170]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)


# In[171]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# ## Logistic Regression

# In[172]:


model = LogisticRegression()


# In[173]:


model.fit(x_train,y_train)
pred = model.predict(x_test)
pred


# In[174]:


y_test


# In[175]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve


# In[176]:


accuracy_score(y_test,pred)


# In[177]:


confusion_matrix(y_test,pred)


# In[178]:


print(classification_report(y_test,pred))


# In[179]:


probability = model.predict_proba(x_test)[:,1]


# In[180]:


probability


# In[181]:


fpr,tpr,threshsholds = roc_curve(y_test,probability)


# In[182]:


plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC CURVE')
plt.show()


# ## Decision Tree

# In[108]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[109]:


y_train


# In[110]:


dt.fit(x_train,y_train)


# In[111]:


pred = dt.predict(x_test)
pred


# In[112]:


y_test


# In[113]:


accuracy_score(y_test,pred)


# In[114]:


confusion_matrix(y_test,pred)


# In[115]:


print(classification_report(y_test,pred))


# In[116]:


probab = dt.predict_proba(x_test)[:,1]
probab


# In[117]:


fpr,tpr,threshsholds = roc_curve(y_test,probab)


# In[118]:


plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('roc curve')
plt.show()


# In[119]:


from sklearn import tree
plt.figure(figsize = (25,25))
tree.plot_tree(dt,filled= True)


# ## Naive Bayes

# In[136]:


from sklearn.naive_bayes import GaussianNB


# In[137]:


clf = GaussianNB()


# In[138]:


clf.fit(x_train, y_train)


# In[139]:


clf.score(x_test,y_test)


# In[140]:


clf.score(x_train,y_train)


# In[141]:


confusion_matrix(y_test,pred)


# In[142]:


print(classification_report(y_test,pred))


# ## Random Forest

# In[125]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[126]:


forest_params = [{'max_depth':list(range(10,15)),'max_features': list(range(0,14))}]


# In[127]:


rf_cv = GridSearchCV(rf,param_grid = forest_params,cv = 10,scoring="accuracy")
rf_cv.fit(x_train,y_train)


# In[128]:


pred = rf_cv.predict(x_test)


# In[129]:


accuracy_score(y_test,pred)


# In[130]:


print(classification_report(y_test,pred))


# In[131]:


rf_cv.best_params_


# In[132]:


rf.fit(x_train,y_train)


# In[133]:


probabo = rf.predict_proba(x_test)[:,1]
probabo


# In[134]:


fpr,tpr,threshsholds = roc_curve(y_test,probabo)


# In[135]:


plt.plot(fpr,tpr)


# In[ ]:




