import csv
import numpy as np
import pandas as pd

means = np.zeros((10, 256))
cov = np.zeros((256, 256))

for i in range(10):
    file = open("handwriting_data/digit_train%d.csv" % i)
    f = csv.reader(file)
    f_array = []
    for row in f:
        row = [float(s) for s in row]
        means[i] = means[i] + np.array(row)
        f_array.append(row)

    means[i] = means[i] / 500
    cov = cov + np.cov(np.array(f_array).T)

cov = cov / 10
inv_cov = np.linalg.inv(cov)
ans = np.zeros((10, 10))

for i in range(10):
    file = open("handwriting_data/digit_test%d.csv" % i)
    f = csv.reader(file)
    for row in f:
        p = np.zeros(10)
        for j in range(len(p)):
            p[j] = np.dot(np.dot(means[j].T, inv_cov), np.array([float(s) for s in row])) - (np.dot(np.dot(means[j].T, inv_cov), means[j])) / 2

        m = p.argmax()
        ans[i][m] = ans[i][m] + 1

ans = ans.T
df = pd.DataFrame({'0': ans[0], '1': ans[1], '2': ans[2], '3': ans[3], '4': ans[4], '5': ans[5], '6': ans[6], '7': ans[7], '8': ans[8], '9': ans[9]}, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
df.style.background_gradient(cmap='winter')

print(df)
