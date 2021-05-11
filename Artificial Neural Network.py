#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network
# 

# ## Importing the libraries

# In[26]:


import numpy as np
import pandas as pd
import tensorflow as tf


# ### Importing the dataset

# In[27]:


dataset = pd.read_csv('Churn_Modelling.csv')


# In[28]:


dataset


# In[30]:


X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


# ### Encoding categorical data
# ### Label Encoding the "Gender" column

# In[31]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print(X)


# ### Label Encoding "Country" column

# In[32]:


labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
print(X)


# ### Replacng with dummy variable

# In[33]:


from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# ### Avoiding dummy variable trap

# In[10]:


##X = X[:, 1:]


# In[37]:


X


# ### Splitting the dataset into the Training set and Test set

# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# ### Feature Scaling

# In[40]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_train=np.asarray(X_train).astype(np.int)
Y_train=np.asarray(Y_train).astype(np.int) 
print(X)


# # Now make ANN

# ## Initializing the ANN (By sequence of layers)

# In[41]:


classifier = tf.keras.models.Sequential()


# #### First layer of ANN (First hidden layer)

# In[42]:


classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Second layer of ANN (Second hidden layer)

# In[43]:


classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Third layer of ANN (Third hidden layer)

# In[44]:


classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))


# #### Output layer of ANN 

# In[45]:


classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# #### compiling the ANN

# In[46]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Fitting the ANN to the training set

# In[49]:


classifier.fit(X_train, Y_train, batch_size = 32, epochs = 100)


# #### Predicting the test result

# In[23]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[24]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


# In[ ]:




