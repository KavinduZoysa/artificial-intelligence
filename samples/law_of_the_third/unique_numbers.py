import numpy as np

l = 1
h = 10000
rand = np.random.randint(low=1, high=h, size=h)
# print(rand)
# rand = np.array([1, 1, 3, 4, 5, 6, 8, 8, 9, 9])
count = 0

for i in range(0, rand.size):
    j = 0
    found = False
    for j in range(0, rand.size):
        if j != i:
            if rand[i] == rand[j]:
                found = True
                break
    if not found:
        count = count + 1

print(count)
