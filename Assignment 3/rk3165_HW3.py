import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#################################################################################################
#Problem-1
x=pd.read_csv('Prob1_X.csv')
y=pd.read_csv('Prob1_y.csv')


class AdaBoost():
    def __init__(self, x, y):
        self.x = x.values
        self.y = y.values
        self.T = 2500
        self.num_of_points = x.shape[0]
        self.weights = np.ones(self.num_of_points) / self.num_of_points
        weight_update = np.zeros(self.num_of_points)
        self.epsilon_list = []
        self.alpha_list = []
        self.training_list = []
        self.upper_bound_list = []
        self.w_list = []
        self.weight_list = []
        self.weight_hat = np.zeros(self.num_of_points)

        self.boost = 0
        self.sum_epsilon = 0

    def pred(self):
        self.y_pred = np.sign(np.dot(self.x, self.w))
        self.epsilon = sum(self.weights[i] * (self.y_pred[i] != self.y[i]) for i in range(self.num_of_points))
        if self.epsilon > 0.5:
            self.w = -self.w
            self.pred()

    def iterate(self):

        # generate the training data
        rand_ind = np.random.choice(self.num_of_points, self.num_of_points, replace=True, p=self.weights)
        x_train = self.x[rand_ind, :]
        y_train = self.y[rand_ind]

        # calculate the weights
        x_x_t = np.linalg.inv(np.dot(x_train.T, x_train))
        self.w = np.dot(np.dot(x_x_t, x_train.T), y_train)

        # predict the data based on the weights
        self.pred()

        # update
        self.epsilon_list.append(self.epsilon)
        alpha = 1 / 2 * np.log((1 - self.epsilon) / self.epsilon)
        self.alpha_list.append(alpha)

        # Training error
        self.boost += self.y_pred * alpha
        boost_pred = np.sign(self.boost)
        training_error = sum(self.y != boost_pred) / self.num_of_points
        self.training_list.append(training_error)

        # upper bound calculation
        self.sum_epsilon += (1 / 2 - self.epsilon) ** 2
        self.upper_bound_list.append(np.exp(-2 * self.sum_epsilon))

        # update weights
        for i in range(self.num_of_points):
            self.weight_hat[i] = self.weights[i] * np.exp(-alpha * self.y[i] * self.y_pred[i])
        self.weight_list.append(self.weights)
        self.weights = self.weight_hat / sum(self.weight_hat)
        self.w_list.append(self.w)

adaboost=AdaBoost(x,y)
T = 2500
for i in range(1, T+1):
    adaboost.iterate()


plt.figure()
plt.ylabel('Error Value')
plt.xlabel('iteration')
plt.title('Tracking of training error and its upper bound across various iterations')
plt.plot(adaboost.training_list, label='empirical training error', color='green')
plt.plot(adaboost.upper_bound_list, label='upper bound', color='red')
plt.grid()
plt.legend()
plt.savefig('1.png')

mean = np.mean(adaboost.weight_list,axis=0)
plt.figure()
plt.title('Empirical Mean for various dimensions')
plt.ylabel('Mean Value')
plt.xlabel('dimensions')
plt.stem(mean, label='weight',markerfmt='C5.')
plt.legend()
plt.grid()
plt.savefig('2.png')

plt.figure()
plt.title('epsilon value across various iterations')
plt.ylabel('epsilon value')
plt.xlabel('iterations')
plt.plot(adaboost.epsilon_list, label='epsilon', color='red')
plt.grid()
plt.legend()
plt.savefig('3.png')

plt.figure()
plt.title('alpha value across various iterations')
plt.ylabel('alpha value')
plt.xlabel('iterations')
plt.plot(adaboost.alpha_list, label='alpha', color='red')
plt.grid()
plt.legend()
plt.savefig('4.png')

################################################################################################################
#Problem-2
def squared_dist(a1,a2):
    dist = np.sum((a1[i]-a2[i])**2 for i in range(len(a1)))
    return np.sqrt(dist)

mu = np.array([[0, 0], [3, 0], [0, 3]])
sigma = np.array([[1, 0], [0, 1]])
pi =[0.2,0.5,0.3]
#randomly group the data
group = np.random.choice(range(3), size=500, p=pi)
#generate 500 gaussain values based on the mean and sigma mentioned above
gaussian = [np.random.multivariate_normal(mu[i], sigma, 500) for i in range(3)]
#produce the random data
data = np.concatenate([gaussian[i][group==i, :] for i in range(3)])


class K_means():
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.num_of_points = data.shape[0]
        self.cluster_num = np.ones(self.num_of_points)
        self.mu_initial = np.random.random((3, self.k, 2))[0]
        self.mu = self.mu_initial
        self.L_fn = []

    def update(self):
        # update the cluster assignment for the given data points
        for i in range(self.num_of_points):
            dist = [squared_dist(self.data[i], self.mu[j]) for j in range(self.k)]
            self.cluster_num[i] = np.argmin(dist)
        # update the mu values for the given cluster assignment
        for i in range(self.k):
            num = sum(self.cluster_num == i)
            self.mu[i] = sum(self.data[j] * (self.cluster_num[j] == i) for j in range(self.num_of_points))
            self.mu[i] = self.mu[i] / num
        # update the loss function
        self.L = 0
        for j in range(self.k):
            temp = sum(
                (self.cluster_num[i] == j) * squared_dist(self.data[i], self.mu[j]) for i in range(self.num_of_points))
            self.L += temp
        self.L_fn.append(self.L)


fig = plt.figure()
plt.xlabel('iterations')
plt.ylabel('Objective function')
plt.grid()
plt.title("Loss function for various values of K in K-means algorithm")
kmeans_data=[]
kmeans_cluster=[]
for k in range(2, 6):
    kmeans = K_means(data,k)
    for i in range(20):
        kmeans.update()
    plt.plot(np.arange(0,20),kmeans.L_fn, label='%d clusters'%k)
    if k in [3,5]:
        kmeans_data.append(kmeans.data)
        kmeans_cluster.append(kmeans.cluster_num)
plt.legend()
plt.savefig('5.png')

#Plot of k=3 data
plt.figure()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("kmeans cluster algorithm with k=3")
data= kmeans_data[0]
cluster=kmeans_cluster[0]
for i in range(3):
    plt.scatter(data[cluster==i,0],data[cluster==i,1],label='cluster'+str(i+1))
plt.legend()
plt.savefig('6.png')

#plot of k=5 data
plt.figure()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("kmeans cluster algorithm with k=5")
data= kmeans_data[1]
cluster=kmeans_cluster[1]
for i in range(5):
    plt.scatter(data[cluster==i,0],data[cluster==i,1],label='cluster'+str(i+1))
plt.legend()
plt.savefig('7.png')

##################################################################################################################################
#Problem 3
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
plt.ylabel('log(L)')
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
plt.ylabel('log(L)')
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




