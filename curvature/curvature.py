import cv2;
import numpy as np;
import math;
from mpmath import *
import matplotlib.pyplot as plt

IM_PATH = "/Users/michaelfilonenko/Downloads/Normal pictures for normal purposes/Снимок экрана 2019-03-27 в 08.32.16.png"

def binary(img):
    binary_img = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 200:
                binary_img[i,j] = 255
            else:
                binary_img[i,j] = 0
    return binary_img;

def find_first_point(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                return (i,j,7)

def go_left(p):
    return (p[1], -p[0])

def go_right(p):
    return (-p[1], p[0])

def sqaure_tracing(img):
    contour = []
    img = binary(img)
    cv2.imshow("binary", img)
    start = find_first_point(img)
    result = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    result[start[0], start[1]] = 255
    next_diff = go_left((1, 0))
    next = (start[0] + next_diff[0], start[1] + next_diff[1])
    while next[0] != start[0] or next[1] != start[1]:
        if img[next[0], next[1]]  == 0:
            next_diff = go_right(next_diff)
        else:
            result[next[0], next[1]] = 255
            contour.append(next)
            next_diff = go_left(next_diff)
        next = (next[0] + next_diff[0], next[1] + next_diff[1])
    cv2.imshow("processing", result)
    cv2.waitKey(0)
    return contour

def curvature(contour):
    curvature = []
    k = 10
    for i in range(len(contour) - k):
        current_point = contour[i]
        forward_point = contour[(i + k) % len(contour)]
        backward_point = contour[(i - k) % len(contour)]
        forward_vector = (current_point[0] - forward_point[0], current_point[1] - forward_point[1])
        backward_vector = (current_point[0] - backward_point[0], current_point[1] - backward_point[1])
        forward_distance = math.sqrt(forward_vector[0] * forward_vector[0] + forward_vector[1] * forward_vector[1])
        backward_distance = math.sqrt(backward_vector[0] * backward_vector[0] + backward_vector[1] * backward_vector[1])
        backward_angle = mp.atan2(abs(backward_point[0] - current_point[0]), abs(backward_point[1] - current_point[1]))
        forward_angle = mp.atan2(abs(forward_point[0] - current_point[0]), abs(forward_point[1] - current_point[1]))
        current_angle = backward_angle / 2 + forward_angle / 2
        forward_delta = forward_angle - current_angle
        curv = forward_delta * (forward_distance + backward_distance) / (2 * forward_distance * backward_distance)
        curvature.append(curv)
    plt.plot(curvature)
    plt.ylabel('curvature')
    plt.show()

if __name__ == "__main__":
    img = cv2.imread(IM_PATH, 0)
    contour = sqaure_tracing(img)
    curvature(contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
