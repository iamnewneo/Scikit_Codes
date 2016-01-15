
# coding: utf-8

# In[3]:

get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# In[4]:

from sklearn.datasets import load_digits
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix


# In[5]:

digits=load_digits()
X = digits.data
y = digits.target


# In[7]:

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=10,criterion='entropy',n_jobs=-1)


# In[8]:

scores = cross_val_score(forest,X,y,cv=10,scoring='accuracy')


# In[10]:

print(scores.mean())


# In[11]:

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
forest_grid=RandomForestClassifier(n_jobs=-1)
criterion_list = ['gini','entropy']
n_estimators_list = list(range(5,15))
max_features_list = ['auto','log2',None]
param_grid = dict(n_estimators=n_estimators_list,max_features=max_features_list,criterion=criterion_list)
grid = GridSearchCV(forest_grid,param_grid,cv=10,scoring='accuracy')


# In[12]:

X.shape


# In[13]:

X=X[:1697]
y=y[:1697]
X_test=X[-100:]
y_test=y[-100:]
X.shape,X_test.shape
grid.fit(X,y)


# In[14]:

#grid.grid_scores_
print(grid.best_score_)
print(grid.best_params_)


# In[15]:

grid.predict(X_test)


# In[16]:

confusion_matrix(grid.predict(X_test),y_test)


# In[17]:

get_ipython().magic('time grid.grid_scores_')


# In[20]:

grid.score


# In[22]:

digits.images[0].shape


# In[ ]:



