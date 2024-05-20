#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


dataset=pd.read_csv('iris (1).data')


# In[3]:


dataset['E'] = dataset['E'].replace(['Iris-setosa'],0.0)
dataset['E'] = dataset['E'].replace(['Iris-versicolor'],1.0)
dataset['E'] = dataset['E'].replace(['Iris-virginica'],2.0)


# In[4]:


dataset


# In[5]:


s=dataset['E']
z=pd.get_dummies(s)
one_hot=z.values
one_hot.shape


# In[6]:


X1=dataset['A']
X2=dataset['B']
X3=dataset['C']
X4=dataset['D']
X1=(X1-X1.mean())/X1.std()
X2=(X2-X2.mean())/X2.std()
X3=(X3-X3.mean())/X3.std()
X4=(X4-X4.mean())/X4.std()
X1=X1.values
X2=X2.values
X3=X3.values
X4=X4.values
X1=X1.reshape((-1,1))
X2=X2.reshape((-1,1))
X3=X3.reshape((-1,1))
X4=X4.reshape((-1,1))


# In[7]:


X=np.hstack((X1,X2,X3,X4)) #maqadir normalize shode
X.shape


# In[8]:


ones=np.ones((150,1))
X=np.hstack((ones,X))
print(X.shape)


# In[9]:


Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,one_hot,test_size=0.2,random_state=123)


# In[10]:


def softmax(z):
    return np.exp(z)/1+(np.sum(np.exp(z)))


# In[15]:


def error(Y,Y_):
    m=Y.shape[0]
    total_error=0.0
    for i in range(m):
        total_error += np.sum(np.max(Y[i]))*(np.log(np.max(Y_[i])))
    #print (type(total_error))   
    return total_error/m
def gradientdescent(X,Y,learning_rate=0.0003,max_steps=500):
    theta =np.random.rand(12).reshape(3,4)
    ones=np.ones((3,1))
    theta=np.hstack((ones,theta)).T
    #theta =np.random.rand(15).reshape(5,3)
    m=X.shape[0]
    #np.zeros((5,1))
    error_list=[]
    s=[]
    for i in range(max_steps):
        h=softmax(X.dot(theta))
        first=(Y-h)
        second=X.T.dot(first)
        second=-second/3
        e=error(Y,h)
        error_list.append(e)
        for i in range(5):
            #for j in range(3):
            theta[i]=theta[i] - learning_rate * second[i]
                #print(theta[i][j])
    #for i in range(m):
     #   s.append(max(Y[i])) 
    #th=np.array(s).reshape((m,1))       
    return theta,error_list 


# In[16]:


new_W,error_list_train=gradientdescent(Xtrain,Ytrain)


# In[17]:


Y_test=softmax(Xtest.dot(new_W))
Y_test.shape


# In[19]:


new_W_,error_list_test=gradientdescent(Xtest,Y_test)


# In[21]:


# visualize
plt.style.use('seaborn')
plt.plot(error_list_test)


# In[29]:


correct=0
m=Ytest.shape[0]
for i in range(m):
    z=np.where(Ytest[i]==max(Ytest[i]))
    t=np.where(Y_test[i]==max(Y_test[i]))
    if z==t:
        correct+=1
    
accuracy=(correct/m)*100    


# In[30]:


print('accuracy=',accuracy)


# In[ ]:




