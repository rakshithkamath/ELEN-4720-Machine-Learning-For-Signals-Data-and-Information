import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# # Problem-2 Classification
#
# #### 1- Naive Bayes,  2- KNN

# In[2]:


X = pd.read_csv("Bayes_classifier\\X.csv", header=None).values
y = pd.read_csv("Bayes_classifier\\y.csv", header=None).values
kf = KFold(n_splits=10, shuffle=True, random_state=0)


# In[3]:


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


# ####  Naive Bayes

# In[4]:


TP = []
FP = []
TN = []
FN = []
lambda0 = []
lambda1 = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    py0 = len(np.where(y_train == 0)[0]) / len(X_train)
    py1 = len(np.where(y_train == 1)[0]) / len(X_train)
    print(py1,py0)

    lambda_0_values = []
    lambda_1_values = []

    for j in range(len(X_train[0])):
        lambda0_numerator = 1
        lambda0_denominator = 1
        lambda1_numerator = 1
        lambda1_denominator = 1

        for i in range(len(X_train)):
            lambda1_numerator += (y_train[i] * X_train[i][j])
            lambda1_denominator += (y_train[i])

            lambda0_numerator += ((1 - y_train[i]) * X_train[i][j])
            lambda0_denominator += (1 - y_train[i])
        lambda_0_values.append((lambda0_numerator / lambda0_denominator)[0])
        lambda_1_values.append((lambda1_numerator / lambda1_denominator)[0])

    lambda0.append(lambda_0_values)
    lambda1.append(lambda_1_values)
    print(lambda1)
    y_predictions = []
    for i in range(len(X_test)):

        y_prob = []
        product_0 = 1
        product_1 = 1

        for j in range(len(X_test[0])):
            poisson_prob_0 = poisson.pmf(X_test[i][j], lambda_0_values[j])
            poisson_prob_1 = poisson.pmf(X_test[i][j], lambda_1_values[j])

            product_0 *= poisson_prob_0
            product_1 *= poisson_prob_1

        y_prob.append(product_0 * py0)
        y_prob.append(product_1 * py1)

        y_predictions.append(np.argmax(y_prob))

    results = compute_confusion_matrix(y_test, y_predictions)

    TP.append(results[0])
    FP.append(results[1])
    FN.append(results[2])
    TN.append(results[3])

# In[5]:


confusion_matrix = [[sum(TP), sum(FP)], [sum(FN), sum(TN)]]
print("The Confusion Matrix is:")

print(confusion_matrix[0])
print(confusion_matrix[1])

print("The Accuracy is:", ((confusion_matrix[0][0] + confusion_matrix[1][1]) /
                           (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] +
                            confusion_matrix[1][1])))

# In[6]:


avg_poisson_parameter_0 = np.mean(lambda0, axis=0)
avg_poisson_parameter_1 = np.mean(lambda1, axis=0)

# In[23]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
ax[0].stem(range(1, 55), avg_poisson_parameter_0, "blue", markerfmt="bo")
ax[0].set_title("Stem plot of Poisson Parameters (Class 0: Not Spam)")
ax[0].set_xlabel("feature number")
ax[0].set_ylabel("value of lambda")
ax[0].set_xticks(np.arange(1, 55, step=1))

ax[1].stem(range(1, 55), avg_poisson_parameter_1, "red", markerfmt="ro")
ax[1].set_title("Stem plot of Poisson Parameters (Class 1: Spam)")
ax[1].set_xlabel("feature number")
ax[1].set_ylabel("value of lambda")
ax[1].set_xticks(np.arange(1, 55, step=1))
plt.show()

# In[21]:


plt.figure(figsize=(15, 5))
plt.stem(range(1, 55), avg_poisson_parameter_0, "blue", markerfmt="bo", label="0 Class: Not Spam")
plt.stem(range(1, 55), avg_poisson_parameter_1, "red", markerfmt="ro", label="1 Class: Spam")
plt.title("Stem plot of Poisson Parameters (seperated by classes)")
plt.xlabel("Feature number")
plt.ylabel("Value of Lambda")
plt.xticks(np.arange(1, 55, step=1))
plt.legend()
plt.show()

# #### KNN