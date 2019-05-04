import multiprocessing as mp
import numpy as np
from multiprocessing import Process, Value, Array, Manager

b = np.array([1, 20, 3, 42, 5, 6, 7, 8, 9, 10, 11])


# [1, 2, 5, 6, 8]

def func(index):
    b[index] = b[index] * 2
    print(b[index])


def func2(n, arr):
    # n.value = 3.1415927
    for index in arr:
        c[index] = c[index] * 2
        print(c[index])
    # for i in arr:
    #     arr[i] = arr[i] * b[2]


# pool = mp.Pool(mp.cpu_count())
# pool.map(func, [index for index in a])
# pool.close()
# pool.join()

# if __name__ == '__main__':
#     num = Value('d', 0.0)
#     c = Array('i', b)
#     a = Array('i', range(5))
#     p = Process(target=func2, args=(c, a))
#     p.start()
#     p.join()
#     print(a[:])
#     print(c[:])

num = Value('d', 0.0)
c = Array('i', b)
a = Array('i', range(5))
# a1 = Array('i', [0, 1, 2])
# a2 = Array('i', [2, 3, 4])
a1 = [0, 1, 2]
a2 = [2, 3, 4]
# p = Process(target=func2, args=(c, a))
# p.start()
# p.join()
# print(a[:])
# print(c[:])

processes = [Process(target=func2, args=(c, a1)), Process(target=func2, args=(c, a2))]

for process in processes:
    process.start()
    print('Process started')

for process in processes:
    process.join()
print(a[:])
print(c[:])
