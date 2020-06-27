#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[3]:


type(digits.data)


# In[4]:


type(digits.target)


# In[5]:


print('Image data shape',digits.data.shape)


# In[6]:


print('Label data shape',digits.target.shape)


# In[7]:


import matplotlib.pyplot as plt 
import numpy as np


# In[8]:


import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
 plt.subplot(1, 10, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


# In[9]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[10]:


def costFunctionReg(theta, X, y,lmbda):
    m = len(y)
    e=10**(-6) 
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta)+e)))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))+e))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)


# In[11]:


def gradRegularization(theta, X, y,lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp


# In[12]:


X = digits.data
Y = digits.target


# In[13]:


m = len(Y)
ones = np.ones((m,1))
X = np.hstack((ones, X))


# In[14]:


(m,n) = X.shape


# In[15]:


m,n


# In[16]:


type(digits.data)


# In[17]:


import scipy


# In[27]:


lmbda = 0.1
k = 10
theta = np.zeros((k,n)) #inital parameters
for i in range(k):
    digit_class = i if i else 10
    theta[i] = scipy.optimize.fmin_cg(f = costFunctionReg, x0 = theta[i],  fprime = gradRegularization, args = (X, (Y == digit_class).flatten(), lmbda), maxiter = 50)


# In[24]:


theta.shape


# In[25]:


theta


# In[28]:


pred = np.argmax(X @ theta.T, axis = 1)

np.mean(pred == Y.flatten()) * 100


# In[ ]:




