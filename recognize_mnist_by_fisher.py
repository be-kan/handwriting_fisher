import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

mnist = datasets.fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
X = X/255.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
means = np.zeros((10, 784))
cov = np.zeros((784, 784))

for i in range(10):
    cov_array = []
    for j in range(len(y_train)):
        if int(y_train[j]) == i:
            means[i] = means[i] + X_train[j]
            cov_array.append(X_train[j])

    cov = cov + np.cov(np.array(cov_array).T) / 10.

means = means / len(y_train)
inv_cov = np.linalg.inv(cov + 0.0000001*np.eye(784))
ans = np.zeros((10, 10))
total = 0

for i in range(10):
    for j in range(len(y_test)):
        if int(y_test[j]) == i:
            p = np.zeros(10)
            for k in range(len(p)):
                p[k] = np.dot(np.dot(means[k].T, inv_cov), X_test[j]) - (np.dot(np.dot(means[k].T, inv_cov), means[k])) / 2

            m = p.argmax()
            ans[m][int(y_test[j])] = ans[m][int(y_test[j])] + 1

df = pd.DataFrame({'0': ans[0], '1': ans[1], '2': ans[2], '3': ans[3], '4': ans[4], '5': ans[5], '6': ans[6], '7': ans[7], '8': ans[8], '9': ans[9]},index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
df.style.background_gradient(cmap='winter')

print(df)
