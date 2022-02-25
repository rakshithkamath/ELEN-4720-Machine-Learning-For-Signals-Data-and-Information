import numpy as np
import pandas as pd
import seaborn as sn
import math
from sklearn.model_selection import KFold
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.special import expit




# #confusion matrix defination
def compute_confusion_matrix(y_test, y_prediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_test)):
        if (y_prediction[i] == y_test[i]):
            if (y_prediction[i] == 0):
                TN += 1
            else:
                TP += 1
        else:
            if (y_prediction[i] == 0):
                FN += 1
            else:
                FP += 1
    return [TP, FP, FN, TN]
#confusion matrix for logistic regression
def compute_confusion_matrix2(y_test, y_prediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_test)):
        if (y_prediction[i] == y_test[i]):
            if (y_prediction[i] == -1):
                TN += 1
            else:
                TP += 1
        else:
            if (y_prediction[i] == -1):
                FN += 1
            else:
                FP += 1
    return [TP, FP, FN, TN]
#Probelm-2
# #main
#
x= pd.read_csv("Bayes_classifier\\X.csv",header = None).values
y=pd.read_csv("Bayes_classifier\\y.csv",header=None).values
# #split the data into 10 parts for test and training
kf = KFold(n_splits=10, shuffle=True, random_state=None)
AvgLambda0=[]
AvgLambda1=[]
TP = []
FP = []
TN = []
FN = []
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pi0=len(np.where(y_train == 0)[0]) / len(X_train)
    pi1=len(np.where(y_train == 1)[0]) / len(X_train)

    lambda0=[]
    lambda1=[]
    for j in range(len(X_train[0])):
        lambda0_n = 1
        lambda0_d = 1
        lambda1_n = 1
        lambda1_d = 1
        for i in range(len(X_train)):
            lambda1_n+= (y_train[i] * X_train[i][j])
            lambda1_d += (y_train[i])

            lambda0_n += ((1 - y_train[i]) * X_train[i][j])
            lambda0_d += (1 - y_train[i])
        lambda0.append((lambda0_n/lambda0_d)[0])
        lambda1.append((lambda1_n/lambda1_d)[0])
    AvgLambda0.append(lambda0)
    AvgLambda1.append(lambda1)
    y_predicted =[]
    for i in range(len(X_test)):

        prod0=1
        prod1=1
        for j in range(len(X_test[0])):
            poisson0=poisson.pmf(X_test[i][j],lambda0[j])
            poisson1= poisson.pmf(X_test[i][j],lambda1[j])
            prod0*=poisson0
            prod1*=poisson1
        y_predicted.append(np.argmax([pi0*prod0,pi1*prod1]))

    result = compute_confusion_matrix(y_test, y_predicted)
    TP.append(result[0])
    FP.append(result[1])
    FN.append(result[2])
    TN.append(result[3])

##plots
#plot of confusion matrix
confusion_Matrix = [[sum(TP), sum(FP)], [sum(FN), sum(TN)]]
print(confusion_Matrix)
df_cm = pd.DataFrame(confusion_Matrix, index = [i for i in ["Email","Spam"]],
                  columns = [i for i in ["Email","Spam"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
#plot of average lambda values of the parameters
avg_poisson_0= np.mean(AvgLambda0,axis=0)
avg_poisson_1=np.mean(AvgLambda1,axis=0)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
ax[0].stem(range(1,55),avg_poisson_0,"blue", markerfmt="bo")
ax[0].set_title("Stem plot of Poisson Parameters (Class 0: Email)")
ax[0].set_xlabel("feature number")
ax[0].set_ylabel("value of lambda")
ax[0].set_xticks(np.arange(1,55, step=1))

ax[1].stem(range(1,55),avg_poisson_1,"red", markerfmt="ro")
ax[1].set_title("Stem plot of Poisson Parameters (Class 1: Spam)")
ax[1].set_xlabel("feature number")
ax[1].set_ylabel("value of lambda")
ax[1].set_xticks(np.arange(1,55, step=1))
plt.show()
############
#part 2 logistic regression and newton method
#appending value 1 to the matrix
x0=np.ones((4600,1))
x=np.hstack((x0,x))
#making the y output 1 or -1
y=np.where(y== 0, -1, y)
stepSize=0.01/4600
TP = []
FP = []
TN = []
FN = []
#split data into test and training
kf = KFold(n_splits=10, shuffle=True, random_state=None)
#part c
AvgObjective_function=[]
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w = np.zeros((55,1))
    max_func=np.zeros((1000,1))
    for i in range (1,1000):
        error = np.dot(X_train.T, ((1 - expit(y_train * np.dot(X_train, w))) * y_train))
        w+=stepSize*error
        max_func[i] = (np.sum(np.log(expit(y_train * np.dot(X_train, w)))))
    max_func=np.delete(max_func, 0)
    AvgObjective_function.append(np.array(max_func))

plt.figure()
for ObjFn in AvgObjective_function:
    plt.plot(np.arange(1, 1000),ObjFn )
plt.plot()
plt.xlabel("iterations")
plt.ylabel("Maximizing function")
plt.title("Plotting objective function over over 1000 iterations with 10 different runs")
plt.show()

#### part d
AvgObjective_function=[]
cross_tab_all=[]
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w = np.zeros((55,1))
    objective_function=0
    objFn=[]
    for i in range(1,101):
       gradient= np.dot(X_train.T, ((1-expit(y_train * np.dot(X_train, w)))* y_train))
       hessian =-np.dot(((expit(y_train * X_train.dot(w)))*(1-expit(y_train * np.dot(X_train, w)))*X_train).T,X_train)
       inv = np.linalg.inv(hessian+ np.diag(([1e-6] * 55)))
       objective_function = (np.sum(np.log(expit(y_train * np.dot(X_train, w)))))
       w-= np.dot(inv,gradient)
       objFn.append(objective_function)

    AvgObjective_function.append(np.array(objFn))
    #Predict Y
    y_pred = expit(X_test.dot(w))
    y_pred[y_pred < 0.5] = -1
    y_pred[y_pred >= 0.5] = 1

    result = compute_confusion_matrix2(y_test, y_pred)
    print(result)
    TP.append(result[0])
    FP.append(result[1])
    FN.append(result[2])
    TN.append(result[3])
#Plots
confusion_Matrix = [[sum(TP), sum(FP)], [sum(FN), sum(TN)]]
print(confusion_Matrix)
df_cm = pd.DataFrame(confusion_Matrix, index = [i for i in ["Email","Spam"]],columns = [i for i in ["Email","Spam"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()


plt.figure()
for ObjFn in AvgObjective_function:
   plt.plot(np.arange(1, 101),ObjFn)
plt.plot()
plt.xlabel("iterations")
plt.ylabel("Maximizing function")
plt.title("Plotting objective function over over 100 iterations with 10 different runs")
plt.show()
##########################################################################################################
#Problem 3
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






