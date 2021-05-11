#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network
# 

# ## Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# ### Importing the dataset

# In[2]:


dataset = pd.read_csv('Churn_Modelling.csv')


# In[3]:


dataset


# In[4]:


X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values
print(X)
print(Y)


# ### Encoding categorical data
# ### Label Encoding the "Gender" column

# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)


# ### One Hot encoding

# In[6]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# ### Feature Scaling

# In[7]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)


# ### Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# # Now make ANN

# ## Initializing the ANN (By sequence of layers)

# In[9]:


classifier = tf.keras.models.Sequential()


# #### First layer of ANN (First hidden layer)

# In[10]:


classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Second layer of ANN (Second hidden layer)

# In[11]:


classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Third layer of ANN (Third hidden layer)

# In[12]:


classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Output layer of ANN 

# In[13]:


classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# #### compiling the ANN

# In[14]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Fitting the ANN to the training set

# In[15]:


classifier.fit(X_train, Y_train, batch_size = 32, epochs = 100)


# #### Predicting the test result

# In[17]:


Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))


# In[18]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


# In[ ]:




