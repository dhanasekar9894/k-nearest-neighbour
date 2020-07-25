#!/usr/bin/env python
# coding: utf-8

# # k-nearest neighbors using sk-learn

# In[1]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:




read_file = pd.read_csv ("C:/Users/dhana/Classifieddata.txt",sep=',')
#read_file.to_csv ("C:/Users/dhana\classifieddatak.csv")
df=read_file.copy()


# In[4]:


df.head()


# In[5]:


#standardize 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[6]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[7]:


scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[8]:


df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # train test split

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30)


# # using knn

# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# In[12]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[13]:


knn.fit(x_train,y_train)


# In[14]:


knn.predict(x_test)


# In[15]:


pred=knn.predict(x_test)


# # predictions and evaluation

# lets evaluate our knn model !!

# In[16]:


from sklearn.metrics import classification_report,confusion_matrix


# In[17]:


print(confusion_matrix(y_test,pred))


# In[18]:


print(classification_report(y_test,pred))


# # choosing k-value

# using elbow method to pick a good k-value

# In[19]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[20]:


print(error_rate)


# In[21]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[22]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)
pred = knn.predict(x_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# now using k=27

# In[23]:


# NOW WITH K=27
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
pred = knn.predict(x_test)

print('WITH K=7')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




