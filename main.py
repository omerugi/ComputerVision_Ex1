import math
from math import exp

import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Open the img file

    if representation == 2:
        image = cv.imread(filename, 1)
        data = np.asarray(image, dtype=np.float32)
        data = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    else:
        image = cv.imread(filename, 0)
        data = np.asarray(image, dtype=np.float32)

    return data


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    k = kernel1[::-1]
    a = np.pad(inSignal, (len(k) - 1, len(k) - 1), 'constant')
    res = np.zeros(len(a) - len(k) + 1)
    for i in range(0, len(a) - len(k) + 1):
        res[i] = np.multiply(a[i:i + len(k)], k).sum()

    return res


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    k = np.flip(kernel2)
    print(inImage.shape)
    a = np.pad(inImage, (k.shape[0] // 2, k.shape[1] // 2), 'edge')
    res = np.ndarray(inImage.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.multiply(a[i:i + k.shape[0], j:j + k.shape[1]], k).sum()

    return res


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    k = np.array([[0, 1, 0],
                  [0, 0, 0],
                  [0, -1, 0]])
    Ix = conv2D(inImage, k.transpose())
    Iy = conv2D(inImage, k)
    mag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    div = np.arctan2(Iy, Ix)
    return mag, div, Ix, Iy


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """

    gaussian = np.ndarray(kernel_size)
    sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
    for x in range(0, kernel_size[0]):
        for y in range(0, kernel_size[1]):
            gaussian[x, y] = math.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (math.pi * (sigma ** 2) * 2)
    return conv2D(in_image, gaussian)


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """

    sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
    shape = (kernel_size[0], kernel_size[1])
    res = cv.GaussianBlur(in_image, shape, sigma)
    return res


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.2) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    s = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])

    thresh *= 255

    my_res = np.sqrt((conv2D(img, s) ** 2 + conv2D(img, s.transpose()) ** 2))
    my = np.ndarray(my_res.shape)
    my[my_res > thresh] = 1
    my[my_res < thresh] = 0
    plt.imshow(my, cmap='gray')
    plt.show()

    cv_res = cv.magnitude(cv.Sobel(img, -1, 1, 0), cv.Sobel(img, -1, 0, 1))
    v = np.ndarray(cv_res.shape)
    v[cv_res > thresh] = 1
    v[cv_res < thresh] = 0

    plt.imshow(v, cmap='gray')
    plt.show()
    return cv_res, my_res


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param I: Input image
    :return: Edge matrix
    """
    img
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])

    d = conv2D(img, k)
    # print(d[0:5, 0:5])
    res = np.zeros(d.shape)

    for i in range(0, d.shape[0]):
        for j in range(0, d.shape[1]):
            try:
                if d[i, j] == 0:
                    if (d[i, j + 1] > 0 and d[i, j - 1] < 0) or (d[i, j + 1] < 0 and d[i, j - 1] > 0) or (
                            d[i + 1, j] > 0 and d[i - 1, j] < 0) or (d[i + 1, j] < 0 and d[i - 1, j] > 0):
                        res[i, j] = 1
                elif d[i, j] > 0:
                    if d[i, j + 1] < 0 or d[i + 1, j] < 0:
                        res[i, j] = 1
                else:
                    if d[i, j + 1] > 0 or d[i + 1, j] > 0:
                        res[i, j] = 1

            except IndexError as e:
                pass
    print("**********************")
    print(res[0:5, 0:5])
    return res


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: :return: Edge matrix
    """
    blur = blurImage2(img, np.array([5, 5]))
    return edgeDetectionZeroCrossingSimple(blur)


def sobleForCanny(img: np.ndarray, ):

    k = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]], dtype=np.float32)
    Ix = cv.filter2D(img, -1, k, borderType=cv.BORDER_REPLICATE)
    Iy = cv.filter2D(img, -1, k.transpose(), borderType=cv.BORDER_REPLICATE)
    mag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    div = np.arctan2(Iy, Ix)
    return mag, div


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """

    mag, div = sobleForCanny(img)
    plt.imshow(mag,cmap='gray')
    plt.show()
    print(np.max(div%180))
    nms = non_max_suppression(mag, div)
    plt.imshow(nms, cmap='gray')
    plt.show()

    for i in range(0, nms.shape[0]):
        for j in range(0, nms.shape[1]):
            try:
                if nms[i][j] <= thrs_2:
                    nms[i][j] = 0
                elif thrs_2 < nms[i][j] < thrs_1:
                    neighbor = nms[i - 1:i + 2, j - 1, j + 2]
                    if neighbor.max() < thrs_1:
                        nms[i][j] = 0
                    else:
                        nms[i][j] = 255
                else:
                    nms[i][j] = 255
            except IndexError as e:
                pass

    cvc = cv.Canny(img.astype(np.uint8), thrs_1, thrs_2)
    return cvc, nms


def non_max_suppression(img: np.ndarray, D: np.ndarray) -> np.ndarray:

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = D % 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


# print(np.convolve([1, 2, 3], [0, 1, 0.5]))
# a = np.array([1,2,3])
# b = np.array([0,1,0.5])
# print(conv1D(a,b))

# arr = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]], dtype=float)
# arr2 = arr.shape
# b = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
#
# b = np.array([1,2,1])
# plt.imshow(cv.filter2D(imReadAndConvert("boxman.jpg", 1), -1, b, borderType=cv.BORDER_REPLICATE))
# plt.show()
# plt.imshow(conv2D(imReadAndConvert("boxman.jpg", 1), b))
# plt.show()

# edgeDetectionSobel(imReadAndConvert("codeMonkey.jpeg", 1))
#
# b = np.array([[0, 1, 0],
#               [1, 0, 1],
#               [0, 1, 0]])
# edgeDetectionZeroCrossingSimple(b)
#
# k = np.ndarray((3, 3)).shape
# plt.imshow(blurImage1(imReadAndConvert("boxman.jpg", 1), np.array([25, 25])))
# plt.show()
# plt.imshow(blurImage2(imReadAndConvert("boxman.jpg", 1), np.array([25, 25])))
# plt.show()

# b = np.array([[0, 0, 0, 0],
#               [0, 1, 0, 0],
#               [0, 0, 0, 0]])
# print(b[0:3, 0:3])
# print(b[1][1])
#
# plt.imshow(edgeDetectionZeroCrossingSimple(imReadAndConvert("codeMonkey.jpeg", 1)), cmap='gray')
# plt.show()
# plt.imshow(edgeDetectionZeroCrossingLOG(imReadAndConvert("codeMonkey.jpeg", 1)), cmap='gray')
# plt.show()

# img = imReadAndConvert("beach (1).jpg",1)
# x_ker = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
# y_ker = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
#
# f, ax = plt.subplots(1, 2)
# edgeX = conv2D(img, x_ker)
# ax[0].imshow(edgeX, cmap="gray")
# edgeX = cv.filter2D(img, -1, x_ker, borderType=cv.BORDER_REPLICATE)
# ax[1].imshow(edgeX, cmap="gray")
# plt.show()

image = cv.imread("boxman.jpg", 0)
data = np.asarray(image, dtype=np.float32)
cvc, myc = edgeDetectionCanny(data, 60, 20)
f, ax = plt.subplots(1, 2)
ax[0].imshow(cvc, cmap="gray")
ax[1].imshow(myc, cmap="gray")
plt.show()

# mag, div, Ix, Iy = convDerivative(imReadAndConvert("frog.png", 1))
# plt.imshow(Ix, cmap='gray')
# plt.show()
# plt.imshow(Iy, cmap='gray')
# plt.show()
# plt.imshow(mag, cmap='gray')
# plt.show()
