import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

X_train= np.array(pd.read_csv('Prob3_Xtrain.csv'))
X_test=np.array(pd.read_csv('Prob3_Xtest.csv'))
y_train=np.array(pd.read_csv('Prob3_ytrain.csv'))
y_test=np.array(pd.read_csv('Prob3_ytest.csv'))

x_train1=[]
x_train0=[]

for i in range(X_train.shape[0]):
    if y_train[i]==1:
        x_train1.append(X_train[i])
    else:
        x_train0.append(X_train[i])

x_train1=np.array(x_train1)
x_train0=np.array(x_train0)
y_train1=np.ones((x_train1.shape[0],1))
y_train0=np.ones((x_train0.shape[0],1))


class EM():

    def __init__(self, data, K):
        self.data = data
        self.k = K
        self.iter = 30
        self.num_of_points = self.data.shape[0]

    def update(self):
        mu_0 = np.array(self.data.mean(axis=0))
        self.cov = [np.cov(self.data.T)] * self.k
        self.mu = np.random.multivariate_normal(mu_0, self.cov[0], self.k)
        self.pi = np.ones(self.k) / self.k
        self.phi = np.zeros((self.k, self.num_of_points))
        self.L = []
        for iterate in range(self.iter):
            # E-step
            for i in range(self.k):
                self.phi[i] = multivariate_normal.pdf(self.data, self.mu[i], self.cov[i], True) * self.pi[i]
            # updating the L function
            self.L.append(np.sum(np.log(self.phi.sum(axis=0))))
            # updating the Phi Values
            self.phi = self.phi / self.phi.sum(axis=0)
            # M-step
            N = self.phi.sum(axis=1)
            self.pi = N / self.num_of_points
            for k in range(self.k):
                self.mu[k] = self.phi[k].dot(self.data) / N[k]
                self.cov[k] = np.multiply(self.phi[k].reshape(-1, 1), (self.data - self.mu[k])).T.dot(
                    self.data - self.mu[k]) / N[k]

EM1 = []
EM0 = []

#Training and showing the objective function for class-1 of the dataset
plt.figure()
plt.xlabel('iteration')
plt.ylabel('L')
plt.grid()
plt.title(' Objective Function for class 1')
for times in range(10):
    em = EM(x_train1, 3)
    em.update()
    plt.plot(np.arange(5,31), em.L[4:], label='run %d' %(times+1))
    EM1.append(em)
plt.legend()
plt.savefig('8.png')

#Training and showing the objective function for class-0 of the dataset
plt.figure()
plt.xlabel('iteration')
plt.ylabel('L')
plt.grid()
plt.title('Objective Function for class 0')
for times in range(10):
    em = EM(x_train0, 3)
    em.update()
    plt.plot(np.arange(5,31), em.L[4:], label='run %d' %(times+1))
    EM0.append(em)
plt.legend()
plt.savefig('9.png')

#Part B
EM1_list = []
EM0_list = []
for i in range(10):
    EM1 = EM(x_train1, 2)
    EM1.update()
    EM1_list.append(EM1)

    EM0 = EM(x_train0, 2)
    EM0.update()
    EM0_list.append(EM0)
best_run_1 = np.argmax([li.L[i] for i, li in enumerate(EM1_list)])
print(best_run_1)


def K_Gaussian_and_Bayes(x_train1,x_train0,y_train1,y_train0,X_test,y_test):
    for K in range(1,5):
        EM1_list=[]
        EM0_list=[]
        for i in range(10):
            EM1 = EM(x_train1, K)
            EM1.update()
            EM1_list.append(EM1)

            EM0 = EM(x_train0, K)
            EM0.update()
            EM0_list.append(EM0)

        best_run_1=np.argmax([li.L[i] for i,li in enumerate(EM1_list)])
        best_run_0=np.argmax([li.L[i] for i,li in enumerate(EM0_list)])

        em1= EM1_list[best_run_1]
        em0= EM0_list[best_run_0]

        #calculate the Bayesian
        phi_1=np.zeros((K,X_test.shape[0]))
        phi_0=np.zeros((K,X_test.shape[0]))

        for i in range(K):
            phi_1[i] = multivariate_normal.pdf(X_test, em1.mu[i], em1.cov[i], True)
            phi_0[i] = multivariate_normal.pdf(X_test, em0.mu[i], em0.cov[i], True)
        y_pred1 = sum(phi_1[k] * em1.pi[k] for k in range(K))
        y_pred0 = sum(phi_0[k] * em0.pi[k] for k in range(K))

        #predict the data and then check the accuracy
        y_pred=[np.argmax([y_pred0[i], y_pred1[i]]) for i in range(X_test.shape[0])]
        accuracy_table=np.zeros((2,2))
        for p in [0, 1]:
                for q in [0, 1]:
                    accuracy_table[p][q] = sum([(y_test[i]==p) & (y_pred[i]==q) for i in range(X_test.shape[0])])
        print(accuracy_table)
        print("Accuracy is %.2f" % ((accuracy_table[0][0] + accuracy_table[1][1]) /X_test.shape[0]))

K_Gaussian_and_Bayes(x_train1,x_train0,y_train1,y_train0,X_test,y_test)