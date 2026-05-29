import numpy as np

#is axes composition possible with the loop notation?

x = np.arange(216).reshape(9, 8, 3)


res = np.zeros((9, 24), dtype=int)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            res[i, k + x.shape[2] * j] = x[i, j, k]

print(res)

res = np.zeros((72, 3), dtype=int)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            res[j + x.shape[1] * i, k] = x[i, j, k]

print(res)

res = np.zeros((216), dtype=int)
x = np.arange(216).reshape(9, 8, 3)

N1 = x.shape[0]
N2 = x.shape[1]
N3 = x.shape[2]

for n1 in range(N1):
    for n2 in range(N2):
        for n3 in range(N3):
            res[n3 + N3 * (n2 + N2 * n1)] = x[n1, n2, n3]

print(res.shape)