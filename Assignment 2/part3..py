import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


X_train = pd.read_csv("Gaussian_process\\X_train.csv",header=None).values
y_train = pd.read_csv("Gaussian_process\\y_train.csv",header=None).values
X_test = pd.read_csv("Gaussian_process\\X_test.csv",header=None).values
y_test = pd.read_csv("Gaussian_process\\y_test.csv",header=None).values


def RootMeanSquareError(y_predicted,y_test):
    y_predicted = np.array(y_predicted)
    y_test = np.array(y_test)
    return np.sqrt(np.sum((y_predicted - y_test)**2)/len(y_test))

def kernel(b,xi,xj):
    kernel = math.exp((-1/b)*np.sum((xi-xj)**2))
    return kernel

def model_fit(X_train,b):
    param =[]
    for i in X_train:
        temp=[]
        for j in X_train:
            temp.append(kernel(b,i,j))
        param.append(temp)
    return np.asarray(param)

def model_predict(X_train,y_train,param,ind,sigma2,b):
    temp =[]
    for j in X_train:
        temp.append(kernel(b,ind,j))
    temp = np.asarray(temp)
    sigma2Identity = sigma2*np.identity(param.shape[0])
    mean = (np.matmul(np.matmul(temp, np.linalg.inv(sigma2Identity + param)), y_train))[0]
    variance = sigma2 + kernel(b,ind,ind) + np.matmul(np.matmul(temp, np.linalg.inv(sigma2Identity + param)),np.transpose(temp))
    return [mean, variance]


####
#Main
#part a
B = [5,7,9,11,13,15]
SigmaSq = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
RMSE=[]
y_test = [i[0] for i in y_test]
for b in B:
    temp =[]
    for sigma in SigmaSq:
        y_predict=[]
        param=model_fit(X_train,b)
        for i in X_test:
            y_predict.append(model_predict(X_train,y_train,param,i,sigma,b)[0])
        temp.append(RootMeanSquareError(y_predict,y_test))
    RMSE.append(temp)
print(RMSE)

## Part c
carWeight_test =X_test[:,3]
carWeight_train = X_train[:,3]
b=5
sigmaSq=2

y_predict=[]
param=model_fit(carWeight_train,b)
for i in carWeight_train:
    y_predict.append(model_predict(carWeight_train, y_train, param, i, sigmaSq, b)[0])
y_predict2 = [i for i,_ in sorted(zip(y_predict,carWeight_train), key=lambda pair: pair[1])]


plt.figure(figsize=(5,5))
plt.scatter(carWeight_train,y_train, label = "True values")
plt.plot(sorted(carWeight_train),y_predict2,color="red",label = "Predicted values")
plt.xlabel("Car Weight")
plt.ylabel("Miles Per Gallon")
plt.title("Miles Per Gallon v/s Car weight")
plt.legend()
plt.show()





