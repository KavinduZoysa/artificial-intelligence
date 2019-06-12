import numpy as np
import time

a = np.random.rand(100000)
b = np.random.rand(100000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print("c = ", c)
print("Vectorization : ", str(1000 * (toc - tic)))

tic = time.time()
for i in range(10000):
    c += a[i] * b[i]
toc = time.time()
print("c = ", c)
print("Non-Vectorization : ", str(1000 * (toc - tic)))
