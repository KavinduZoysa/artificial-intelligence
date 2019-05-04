import numpy as np
import time
import cv2 as cv
import time
import multiprocessing as mp
from multiprocessing import Process, Value, Array, Manager

zone1_template = cv.imread('./image_templates/zone1.bmp', 0)
height_z, width_z = zone1_template.shape

a = []
i = 0
kernal_half_width = 2
for h in range(kernal_half_width, height_z - kernal_half_width):
    for w in range(kernal_half_width, width_z - kernal_half_width):
        if zone1_template[h, w] == 255:
            a.append([h, w])

# # Recreate the image
# test_image = np.zeros((480, 720))
# for i in range(a.__len__()):
#     test_image[a[i][0], a[i][1]] = 255
#     print(i)
#
# cv.imshow('image2', test_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Read the sample image
image_name = '/home/kavindu/MSc/Road Traffic Project/code/input_images/kasbawa_720p/image111736.bmp'
img = cv.imread(image_name, 0)
edges = cv.Canny(img, 100, 200)
zone1_temp_img = cv.bitwise_and(zone1_template, edges, mask=None)

# Get the edges from a no vehicles image
no_vehicles_img = cv.imread('./image_templates/novehicles.bmp', 0)
no_vehicles_edges = cv.Canny(no_vehicles_img, 100, 200)
# Get the zone1 image
no_vehicles_zone1_img = cv.bitwise_and(zone1_template, no_vehicles_edges, mask=None)

height, width = zone1_temp_img.shape
image = np.zeros((height, width))
kernel = np.ones((kernal_half_width * 2 + 1, kernal_half_width * 2 + 1))


def determine_edge(point):
    h = point[0]
    w = point[1]
    if no_vehicles_zone1_img[h, w] == 0 and zone1_temp_img[h, w] == 255:
        # Create a kernel from the image
        image_kernel = no_vehicles_zone1_img[h - kernal_half_width:h + kernal_half_width + 1,
                       w - kernal_half_width:w + kernal_half_width + 1]
        if np.sum(np.multiply(image_kernel, kernel)) <= 0:
            print(h, w)
            image[h, w] = 255
            return point


def determine_edge_v2(arr, imgx):
    for i in arr:
        h = i[0]
        w = i[1]
        if no_vehicles_zone1_img[h, w] == 0 and zone1_temp_img[h, w] == 255:
            # Create a kernel from the image
            image_kernel = no_vehicles_zone1_img[h - kernal_half_width:h + kernal_half_width + 1,
                           w - kernal_half_width:w + kernal_half_width + 1]
            if np.sum(np.multiply(image_kernel, kernel)) <= 0:
                # print(h, w)
                imgx[h][w] = 255.0


# for i in range(a.__len__()):
#     h = a[i][0]
#     w = a[i][1]
#     if no_vehicles_zone1_img[h, w] == 0 and zone1_temp_img[h, w] == 255:
#         # Create a kernel from the image
#         image_kernel = no_vehicles_zone1_img[h - kernal_half_width:h + kernal_half_width + 1,
#                        w - kernal_half_width:w + kernal_half_width + 1]
#         if np.sum(np.multiply(image_kernel, kernel)) <= 0:
#             image[h, w] = 255


# pool = mp.Pool(mp.cpu_count())
# result = pool.map(determine_edge, [point for point in a])
# pool.close()
# pool.join()


# a0 = Array('a', a)
# img = Array('i', [[1, 2]])
start = time.time()
img = Manager().list(image.tolist())
a1 = a[0: int(a.__len__()/5)]
a2 = a[int(a.__len__()/5): int(a.__len__()*2/5)]
a3 = a[int(a.__len__()*2/5): int(a.__len__()*3/5)]
a4 = a[int(a.__len__()*3/5): int(a.__len__()*4/5)]
a5 = a[int(a.__len__()*4/5): int(a.__len__()*5/5)]
p1 = Process(target=determine_edge_v2, args=(a1, img))
p2 = Process(target=determine_edge_v2, args=(a2, img))
p3 = Process(target=determine_edge_v2, args=(a3, img))
p4 = Process(target=determine_edge_v2, args=(a4, img))
p5 = Process(target=determine_edge_v2, args=(a5, img))
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()


i = np.array(img)

print("edges : ", cv.countNonZero(i))
print("Time : ", time.time() - start)
cv.imshow('image2', i)
cv.waitKey(0)
cv.destroyAllWindows()
