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
    a = np.pad(inImage, (k.shape[0] // 2, k.shape[1] // 2), 'edge')
    res = np.ndarray(inImage.shape)
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            res[i, j] = np.multiply(a[i:i + k.shape[0], j:j + k.shape[1]], k).sum()

    return res


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    k = np.array([[1],
                  [0],
                  [-1]])
    Ix = conv2D(inImage, k.transpose())
    Iy = conv2D(inImage, k)
    mag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    div = np.arctan(Ix / Iy)
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
    shape=(kernel_size[0],kernel_size[1])
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

    cv_res = cv.magnitude(cv.Sobel(img, -1, 1, 0), cv.Sobel(img, -1, 0, 1, ))
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
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])
    img_dervative = conv2D(img, k)
    print(img_dervative)
    zero_crossings = np.where(np.diff(np.sign(img_dervative)))
    print(zero_crossings)

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: :return: Edge matrix
    """
    kernel = np.ndarray((3, 3))
    b_img = blurImage1(img, kernel)
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])
    img_dervative = conv2D(img, k)


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """


# print(np.convolve([1, 2, 3], [0, 1, 0.5]))
# a = np.array([1,2,3])
# b = np.array([0,1,0.5])
# print(conv1D(a,b))

# arr = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]], dtype=float)
# arr2 = arr.shape
# b = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
#
b = np.array([[0,0,0],
             [0,1,0],
             [0,0,0]])
plt.imshow(cv.filter2D(imReadAndConvert("boxman.jpg", 1),-1,b,borderType=cv.BORDER_REPLICATE))
plt.show()
plt.imshow(conv2D(imReadAndConvert("boxman.jpg", 1),b))
plt.show()

# edgeDetectionSobel(imReadAndConvert("boxman.jpg", 1))

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
