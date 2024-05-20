import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas as pd

class Classifier:
  def __init__(self):
      with open('iris.data') as f:
        content = [line.strip().split(',') for line in f] 

      temp = [[float(string) for string in inner[:4]] for inner in content]
      temp = [[1] + x for x in temp]
      X = np.array( temp )
      label = []
      for row in content:
        if row[4] == 'Iris-setosa':
          label.append(1)
        elif row[4] == 'Iris-versicolor':
          label.append(2)
        elif row[4] == 'Iris-virginica':
          label.append(3)
        else:
          label.append(4)

      mean = np.mean(X, axis=0)
      std = np.std(X, axis=0)

      for i in range(1, len(X)):
        X[i][1:] = (X[i][1:] - mean[1:]) / std[1:]

      max1 = np.max(X, axis=0)
      min1 = np.min(X, axis=0)
      for i in range(1, len(X)):
        X[i][1:] = (X[i][1:] - min1[1:]) / (max1[1:] - min1[1:])   
      
      self.xTrain = np.concatenate((np.concatenate((X[0:40] , X[50:90]), axis=0) , X[100:140]), axis=0)
      self.labelTrain = np.concatenate((np.concatenate((label[0:40] , label[50:90]), axis=0) , label[100:140]), axis=0)  
      self.xTest = np.concatenate((np.concatenate((X[40:50] , X[90:100]), axis=0) , X[140:150]), axis=0)  
      self.labelTest = np.concatenate((np.concatenate((label[40:50] , label[90:100]), axis=0) , label[140:150]), axis=0) 

  def sigmoid(self,z):
    return 1 / (1 + np.exp(-z))

  def gradient_ascent(self, X, y, teta, alpha, iterations, epsilon):
    preTeta = teta
    cost = []
    itr = []
    for i in range(iterations):
      cost.append( (0.5 * ( self.sigmoid(X @ teta) - y ).T  @  ( self.sigmoid(X @ teta) - y ))[0][0] )
      itr.append(i)
      teta = teta + ((alpha) * (X.T @ ( y - self.sigmoid(X @ teta) )))
      diffTeta = teta - preTeta
      normDiffTeta = np.linalg.norm(diffTeta)
      # if normDiffTeta < epsilon:
      #   return teta, cost, itr
      preTeta = teta
    return teta, cost, itr

  def vote(self, x):
    vote = []
    if (x @ self.teta12) < 0:
      vote.append(1)
    else:
      vote.append(2)

    if (x @ self.teta13) < 0:
      vote.append(1)
    else:
      vote.append(3)

    if (x @ self.teta23) < 0:
      vote.append(2)
    else:
      vote.append(3)

    return max(set(vote), key=vote.count)

  def probability(self, x):
    prob = []
    prob.append(x @ self.teta1all)
    prob.append(x @ self.teta2all)
    prob.append(x @ self.teta3all)
    # print(prob)
    return prob.index(min(prob)) + 1

  def oneVsOne(self):

    y = np.concatenate(([[0]] * 40 , [[1]] * 40), axis=0)
    x12 = self.xTrain[:80]
    x13 = np.concatenate(( self.xTrain[:40], self.xTrain[80:]), axis=0)
    x23 = np.concatenate(( self.xTrain[40:80], self.xTrain[80:]), axis=0)

    maxIteration = 100
    rate = 0.03
    epsilon = 0.01
    n = np.size(x12,1)
    initialTeta = np.zeros((n,1))

    self.teta12, _, _ = self.gradient_ascent(x12, y[:], initialTeta, rate, maxIteration, epsilon)
    self.teta13, _, _ = self.gradient_ascent(x13, y[:], initialTeta, rate, maxIteration, epsilon)
    self.teta23, _, _ = self.gradient_ascent(x23, y[:], initialTeta, rate, maxIteration, epsilon)

    # print("teta12", self.teta12)
    # print("teta13", self.teta13)
    # print("teta23", self.teta23)

    currectTrain = 0
    for i in range(len(self.xTrain)):
      classi = self.vote(self.xTrain[i])
      if (classi == self.labelTrain[i]):
        currectTrain += 1

    accuracyTrain = currectTrain / len(self.xTrain)

    currectTest = 0
    for i in range(len(self.xTest)):
      classi = self.vote(self.xTest[i])
      if (classi == self.labelTest[i]):
        currectTest += 1

    accuracyTest = currectTest / len(self.xTest)   

    print("one vs one train accuracy", accuracyTrain*100,"%")
    print("one vs one test accuracy", accuracyTest*100,"%")

  def oneVsAll(self):
    s0 = [[0]] * 40
    s1 = [[1]] * 40
    y1all = np.array(s0 + s1 + s1)
    y2all = np.array(s1 + s0 + s1)
    y3all = np.array(s1 + s1 + s0)
    maxIteration = 100
    rate = 0.03
    epsilon = 0.01
    n = np.size(self.xTrain,1)
    initialTeta = np.zeros((n,1))

    self.teta1all, cost1all, iteration1all = self.gradient_ascent(self.xTrain[:], y1all, initialTeta, rate, maxIteration, epsilon)
    self.teta2all, cost2all, iteration2all = self.gradient_ascent(self.xTrain[:], y2all, initialTeta, rate, maxIteration, epsilon)
    self.teta3all, cost3all, iteration3all = self.gradient_ascent(self.xTrain[:], y3all, initialTeta, rate, maxIteration, epsilon)

    # print("teta1", self.teta1all)
    # print("teta2", self.teta2all)
    # print("teta3", self.teta3all)

    currectTrain = 0
    for i in range(len(self.xTrain)):
      classi = self.probability(self.xTrain[i])
      if (classi == self.labelTrain[i]):
        currectTrain += 1

    accuracyTrain = currectTrain / len(self.xTrain)

    currectTest = 0
    for i in range(len(self.xTest)):
      classi = self.probability(self.xTest[i])
      if (classi == self.labelTest[i]):
        currectTest += 1

    accuracyTest = currectTest / len(self.xTest)   

    print("one vs all train accuracy", accuracyTrain*100,"%")
    print("one vs all test accuracy", accuracyTest*100,"%")

    plt.plot(iteration1all, cost1all, color = "#9933ff")
    plt.plot(iteration2all, cost2all, color = "#ff4d4d")
    plt.plot(iteration3all, cost3all, color = "#00ace6")
    plt.show()

classifier = Classifier()
classifier.oneVsOne()
classifier.oneVsAll()
