import numpy as np
from time import time
import multiprocessing as mp

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()


def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


# # Solution Without Paralleization
results = []
start = time()
for row in data:
    results.append(howmany_within_range(row, minimum=0, maximum=1))
print("Without Paralleization : ", time() - start)
print(results[:15])

# Parallelizing using Pool.apply()
# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())
start = time()
# Step 2: `pool.apply` the `howmany_within_range()`
results1 = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
print("Parallelizing using Pool.apply() : ", time() - start)
# Step 3: Don't forget to close
pool.close()
print(results1[:15])


# Parallelizing using Pool.map()
# Redefine, with only 1 mandatory argument.
def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


pool = mp.Pool(mp.cpu_count())
start = time()
results = pool.map(howmany_within_range_rowonly, [row for row in data])
print("Parallelizing using Pool.map() : ", time() - start)
pool.close()
print(results[:15])
