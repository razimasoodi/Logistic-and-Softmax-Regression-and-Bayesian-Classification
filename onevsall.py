#!/usr/bin/env python
# coding: utf-8

# In[374]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


dataset=pd.read_csv('iris (1).data')
dataset1=dataset.copy(deep=True)
dataset2=dataset.copy(deep=True)


# In[3]:


dataset['E'] = dataset['E'].replace(['Iris-setosa'],1.0)
dataset['E'] = dataset['E'].replace(['Iris-versicolor'],0.0)
dataset['E'] = dataset['E'].replace(['Iris-virginica'],0.0)
dataset


# In[4]:


X1=dataset['A']
X2=dataset['B']
X3=dataset['C']
X4=dataset['D']
Y1=dataset['E']
X1=(X1-X1.mean())/X1.std()
X2=(X2-X2.mean())/X2.std()
X3=(X3-X3.mean())/X3.std()
X4=(X4-X4.mean())/X4.std()
X1=X1.values
X2=X2.values
X3=X3.values
X4=X4.values
Y1=Y1.values
X1=X1.reshape((-1,1))
X2=X2.reshape((-1,1))
X3=X3.reshape((-1,1))
X4=X4.reshape((-1,1))
Y1=Y1.reshape((-1,1))


# In[5]:


X=np.hstack((X1,X2,X3,X4)) #maqadir normalize shode


# In[6]:


X.shape


# In[7]:


ones=np.ones((150,1))
X=np.hstack((X,ones))
X.shape


# In[8]:


Xtrain1,Xtest1,Ytrain1,Ytest1=train_test_split(X,Y1,test_size=0.2,random_state=123)


# In[9]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[399]:



def error(X,Y,Y_):
    m=X.shape[0]
    total_error=0.0
    for i in range(m):
        total_error += ( Y_[i]- Y[i])**2
        
    return total_error/m     
     

def gradientascent(X,Y,learning_rate=0.03,max_steps=100):
    theta =np.random.rand(5).reshape(5,1)
    #np.zeros((5,1))
    error_list=[]
    for i in range(max_steps):
        #grad=gradiant(X,Y,theta)
        h=sigmoid(np.dot(X, theta))
        first=np.dot(X.T,((Y-h)))
        #print(first)
        e=error(X,Y,h)
        #print(e)
        error_list.append(e)
        for i in range(5):
            theta[i]=theta[i] + learning_rate * first[i]
        
    return theta,error_list 


# In[400]:


new_W1,error_list_train1=gradientascent(Xtrain1,Ytrain1)


# In[401]:


Y_test1 = sigmoid(Xtest1.dot(new_W1))


# In[402]:


accuracy_score(Ytest1, Y_test1.round() ,normalize=True)


# In[403]:


new_W_,error_list_test1=gradientascent(Xtest1,Y_test1)


# In[404]:


plt.plot(error_list_test1)


# In[405]:


dataset1['E'] = dataset1['E'].replace(['Iris-setosa'],0.0)
dataset1['E'] = dataset1['E'].replace(['Iris-versicolor'],1.0)
dataset1['E'] = dataset1['E'].replace(['Iris-virginica'],0.0)
dataset1


# In[406]:


Y2=dataset1['E']
Y2=Y2.values
Y2=Y2.reshape((-1,1))
Y2.shape  


# In[407]:


Xtrain2,Xtest2,Ytrain2,Ytest2=train_test_split(X,Y2,test_size=0.2,random_state=123)


# In[408]:


new_W2,error_list_train2=gradientascent(Xtrain2,Ytrain2)


# In[409]:


Y_test2 = sigmoid(Xtest2.dot(new_W2))


# In[410]:


accuracy_score(Ytest2, Y_test2.round() ,normalize=True)


# In[411]:


new_W_,error_list_test2=gradientascent(Xtest2,Y_test2)


# In[412]:


plt.plot(error_list_test2)


# In[413]:


dataset2['E'] = dataset2['E'].replace(['Iris-setosa'],0.0)
dataset2['E'] = dataset2['E'].replace(['Iris-versicolor'],0.0)
dataset2['E'] = dataset2['E'].replace(['Iris-virginica'],1.0)
dataset2


# In[414]:


Y3=dataset2['E']
Y3=Y3.values
Y3=Y3.reshape((-1,1))
Y3.shape


# In[415]:


Xtrain3,Xtest3,Ytrain3,Ytest3=train_test_split(X,Y3,test_size=0.2,random_state=123)


# In[416]:


new_W3,error_list_train3=gradientascent(Xtrain3,Ytrain3)


# In[417]:


Y_test3 = sigmoid(Xtest3.dot(new_W3))


# In[418]:


accuracy_score(Ytest3, Y_test3.round() ,normalize=True)


# In[419]:


new_W_,error_list_test3=gradientascent(Xtest3,Y_test3)


# In[420]:


plt.plot(error_list_test3)


# In[421]:


plt.style.use('seaborn')
plt.plot(error_list_test1,label='model1')
plt.plot(error_list_test2,color='red',label='model2')
plt.plot(error_list_test3,color='green',label='model3')
plt.title('cost curve')
plt.legend()
plt.show()


# In[422]:


error1=np.array(error_list_test1)
error2=np.array(error_list_test2)
error3=np.array(error_list_test3)


# In[423]:


for i in range(100):
    Total_error=error1+error2+error3
    Total_error/3


# In[424]:


plt.style.use('seaborn')
plt.plot(Total_error)
plt.title('the avg of cost curve')


# In[ ]:





# In[ ]:




