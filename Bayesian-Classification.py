import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas as pd
from scipy.stats import multivariate_normal

def normalize_rows(x):
  return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

class Bayesian:
  def __init__(self,name):

    f = open(name)
    X = []
    Y = []
    self.sumClass0 = [0,0]
    self.sumClass1 = [0,0]
    self.countClass1 = 0
    self.countClass0 = 0
    self.countData = 0

    for line in f:
      self.countData += 1
      data_line = line.rstrip().split(',')
      features = []
      features.append(float(data_line[0]))
      features.append(float(data_line[1]))
      X.append(features)
      Y.append([float(data_line[2])])

      if float(data_line[2]) == 1:
        self.sumClass1[0] += features[0]
        self.sumClass1[1] += features[1]
        self.countClass1 += 1
      elif float(data_line[2]) == 0:
        self.sumClass0[0] += features[0]
        self.sumClass0[1] += features[1]
        self.countClass0 += 1

    self.X = X
    self.Y = Y
    self.npX = np.array(X)
    self.npY = np.array(Y)
  
  def readTest(self, name):
    f = open(name)
    self.xTest = []
    self.yTest = []

    for line in f:
      data_line = line.rstrip().split(',')
      features = []
      features.append(float(data_line[0]))
      features.append(float(data_line[1]))
      self.xTest.append(features)
      self.yTest.append([float(data_line[2])])

  def findClassifierParam(self):

    self.meanClass0 = np.array( [element / self.countClass0 for element in self.sumClass0] )
    self.meanClass1 = np.array( [element / self.countClass1 for element in self.sumClass1] )
    #print('self.meanClass1',self.meanClass1)

    self.phiClass0 = self.countClass0 / self.countData
    self.phiClass1 = self.countClass1 / self.countData

    demeanX = []
    for i in range(self.countData):
      
      temp = []
      if self.Y[i][0] == 1:
        temp.append(self.X[i][0] - self.meanClass1[0])
        temp.append(self.X[i][1] - self.meanClass1[1])

      elif self.Y[i][0] == 0:
        temp.append(self.X[i][0] - self.meanClass0[0])
        temp.append(self.X[i][1] - self.meanClass0[1])

      demeanX.append(temp)

    demeanX = np.array(demeanX)
    demeanX = demeanX/np.linalg.norm(demeanX, ord=2, axis=1, keepdims=True)

    self.cov = np.round( (1/self.countData) * (demeanX.T @ demeanX) , 5)
    
    print("mean class 0", self.meanClass0)
    print("mean class 1", self.meanClass1)
    print("cov ", self.cov)
  
  def classifier(self, n, dataX, dataY):
    estimate = []
    coefficient = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.cov))) )
    correct = 0
    for i in range(len(dataX)):
      
      demean0 = dataX[i] - self.meanClass0
      demean1 = dataX[i] - self.meanClass1
      
      covInvers = np.linalg.inv(self.cov)
      pXY0 = coefficient * (np.exp(  -0.5 * ( (demean0.T @ covInvers) @ demean0 )  ))
      pXY1 = coefficient * (np.exp(  -0.5 * ( (demean1.T @ covInvers) @ demean1 )  ))
      #print('cov invers=',pXY0)
      l0 = pXY0 * self.phiClass0
      l1 = pXY1 * self.phiClass1
      
      if (l0 > l1):
        estimate.append(0)
        if dataY[i][0] == 0:
          correct += 1
      else:
        estimate.append(1)
        if dataY[i][0] == 1:
          correct += 1
    #print('self.cov',estimate)
    accuracy = correct / len(dataX) * 100

    print(accuracy, "%")

  def plotDecisionBoundary(self):

    class1 = []
    class0 = []
    for i in range(len(self.Y)):
      if self.Y[i][0] == 1:
        class1.append(self.X[i])
      elif self.Y[i][0] == 0:
        class0.append(self.X[i])
    
    #train
    plt.scatter(*zip(*class0), color = "#ff4d4d", label = 'train class 0') # y = 0, train
    plt.scatter(*zip(*class1), color = "#80dfff", label = 'train class 1') # y = 1, train

    class1 = []
    class0 = []
    for i in range(len(self.yTest)):
      if self.yTest[i][0] == 1:
        class1.append(self.xTest[i])
      elif self.yTest[i][0] == 0:
        class0.append(self.xTest[i])

    #test
    plt.scatter(*zip(*class0), color = "#ff8080")  # y = 0, test
    plt.scatter(*zip(*class1), color = "#00ace6")  # y = 1, test

    self.a = (np.linalg.inv(self.cov) @ (self.meanClass1 - self.meanClass0))
    self.b = (0.5 * (self.meanClass0.T @ np.linalg.inv(self.cov)) @ self.meanClass0) 
    - (0.5 * (self.meanClass1.T @ np.linalg.inv(self.cov)) @ self.meanClass1) 
    + np.log(self.phiClass0/self.phiClass1)

    #a[0] * x[0] + a[1] * x[1] + b = 0    => x1 = -(b + x0 * a[0]) / a[1]
    x0 = np.array([row[0] for row in self.npX])
    x1 = -(self.b + x0 * self.a[0]) / self.a[1]
    plt.plot(x0, x1, color = "#9933ff")
    plt.show()

  def plotPDF(self):
    x, y = np.mgrid[-3.0:10.0:30j, -7.5:12.5:30j]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])
    mean0 = np.array(self.meanClass0)
    mean1 = np.array(self.meanClass1)

    z0 = multivariate_normal.pdf(xy, mean=mean0, cov=self.cov)
    z1 = multivariate_normal.pdf(xy, mean=mean1, cov=self.cov)
    # Reshape back to a (30, 30) grid.
    z0 = z0.reshape(x.shape)
    z1 = z1.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x,y,z1,color = 'red' , label = 'class 2')
    ax.plot_wireframe(x,y,z0 , label= 'class 1')
    plt.show()

    plt.contour(x, y, z0, colors='blue')
    plt.contour(x, y, z1, colors='red')
    x0 = np.array([row[0] for row in self.npX])
    x1 = -(self.b + self.a[0]*x0)/self.a[1]
    plt.plot(x0, x1, c='k') 
    plt.show()


b = Bayesian('BC-Train1.csv')
b.findClassifierParam()
print("train1 accuracy: ")
b.classifier(n=2, dataX=b.npX, dataY=b.npY) 
b.readTest('BC-Test1.csv')
print("test1 accuracy: ")
b.classifier(n=2, dataX=b.xTest, dataY=b.yTest)
b.plotDecisionBoundary()
b.plotPDF()

b2 = Bayesian('BC-Train2.csv')
b2.findClassifierParam()
print("train2 accuracy: ")
b2.classifier(n=2, dataX=b2.npX, dataY=b2.npY) 
b2.readTest('BC-Test2.csv')
print("test2 accuracy: ")
b2.classifier(n=2, dataX=b2.xTest, dataY=b2.yTest)
b2.plotDecisionBoundary()
b2.plotPDF()
