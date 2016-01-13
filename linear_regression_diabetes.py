
# coding: utf-8

# In[61]:

get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# In[62]:

diabetes=load_diabetes()
diabetes.data.shape,diabetes.target.shape


# In[114]:

X=diabetes.data[:, np.newaxis, 5]
y=diabetes.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)
y_pred=linear_reg.predict(X_test)
print(y_test[:11])
print(y_pred[:11])
mean_error=np.mean(y_test-y_pred)
print(mean_error)


# In[92]:

plt.scatter(X_test,y_test,cmap='rainbow')
plt.plot(X_test,y_pred)


# In[119]:

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[129]:

X=diabetes.data[:,np.newaxis,5]
y=diabetes.target
X_train_new,X_test_new,y_train_new,y_test_new=train_test_split(X,y,random_state=1)
model=make_pipeline(PolynomialFeatures(2),LinearRegression())
model.fit(X_train_new,y_train_new)
y_pred_new=model.predict(X_test)


# In[130]:

plt.scatter(X_test_new,y_test_new,cmap='rainbow')
plt.plot(X_test_new,y_pred_new)


# In[131]:

mean_error=np.mean(y_test_new-y_pred_new)
print(mean_error)


# In[ ]:



