# import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
import time
import os


# def square(x):
#     # calculate the square of the value of x
#     return x * x
#
#
# if __name__ == '__main__':
#     # Define the dataset
#     dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#
#     # Output the dataset
#     print('Dataset: ' + str(dataset))
#
#     # Run this with a pool of 5 agents having a chunksize of 3 until finished
#     agents = 5
#     chunksize = 3
#     start = time.time()
#     with Pool(processes=agents) as pool:
#         result = pool.map(square, dataset, chunksize)
#     end = time.time()
#     # Output the result
#     print('Result:  ' + str(result))
#     print('Time: ', end - start)
#
#     start = time.time()
#     for i in range(dataset.__len__()):
#         square(dataset[i])
#     end = time.time()
#     print('Time: ', end - start)


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    info('function f')
    print('hello', name)


if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
