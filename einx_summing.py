import einx
import numpy as np


x = np.arange(120).reshape(5, 2, 3, 4)
z = einx.sum("a [b] c d -> a c d", x)

print(z)

y = einx.sum("a [b] c d", x)

print(y)


res = np.zeros((5, 3, 4), dtype=int)
for i in range(x.shape[0]):
    for j in range(x.shape[2]):
        for k in range(x.shape[3]):
            res[i, j, k] = sum(x[i, :, j, k])

print(res)

"""
v = einx.sum("a [b] c", y)

print(v)
"""