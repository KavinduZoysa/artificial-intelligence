import numpy as np
import multiprocessing

cores = multiprocessing.cpu_count()

low = 0
high = 10000
rand = np.random.randint(low=low, high=high, size=high)


def get_count(numbers):
    count = 0
    for i in numbers:
        for element in rand:
            if i == element:
                count = count + 1
                break
    return count


def split_array(size, arr):
    array = []
    for i in range(size):
        array.append(arr[int(i * arr.size / cores):int((i + 1) * arr.size / cores)])
    return array


process = multiprocessing.Pool(cores)

arr = np.arange(high)
split_rand = split_array(cores, arr)

x1 = process.map(get_count, [i for i in split_rand])
x2 = np.sum(x1)
print("Not appeared count : ", high - x2 - 1)

