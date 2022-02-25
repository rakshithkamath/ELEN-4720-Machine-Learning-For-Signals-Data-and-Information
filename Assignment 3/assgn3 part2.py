import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

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


