import ex2_utils
import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt


def check_conv1():
    a = np.arange(10000.0)
    b = np.array([0, 1, 0.5])
    print(np.sum(np.convolve(a, b) - ex2_utils.conv1D(a, b)))


def check_conv2(img: np.ndarray):
    kernel = np.array([[1], [0], [1]])
    ans1 = ex2_utils.conv2D(img, kernel)
    ans2 = cv.filter2D(img.astype(np.float32), -1, kernel.astype(np.float32), borderType=cv.BORDER_REPLICATE).astype(
        np.float32)
    f, ax = plt.subplots(1, 2)
    f.suptitle('conv2d', fontsize=16)
    ax[0].imshow(ans1, cmap="gray")
    ax[0].set_title('mine')
    ax[1].imshow(ans2, cmap="gray")
    ax[1].set_title('cv')
    plt.show()


def check_Derivative(img: np.ndarray):
    div, mag, Ix, Iy = ex2_utils.convDerivative(img)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.suptitle('Derivative', fontsize=16)
    ax1.imshow(Ix, cmap="gray")
    ax1.set_title('Ix')
    ax2.imshow(Iy, cmap="gray")
    ax2.set_title('Iy')
    ax3.imshow(mag, cmap="gray")
    ax3.set_title('mag')
    ax4.imshow(div, cmap="gray")
    ax4.set_title('div')
    plt.show()


def check_blur(img: np.ndarray):
    size = 23
    ans1 = ex2_utils.blurImage1(img, size)
    ans2 = ex2_utils.blurImage2(img, size)
    f, ax = plt.subplots(1, 2)
    f.suptitle('blur', fontsize=16)
    plt.title("blur")
    ax[0].imshow(ans1, cmap="gray")
    ax[0].set_title('mine')
    ax[1].imshow(ans2, cmap="gray")
    ax[1].set_title('cv')
    plt.show()


def check_sobel(img: np.ndarray):
    ans1, ans2 = ex2_utils.edgeDetectionSobel(img)
    f, ax = plt.subplots(1, 2)
    f.suptitle('sobel', fontsize=16)
    ax[0].imshow(ans1, cmap="gray")
    ax[0].set_title('cv')
    ax[1].imshow(ans2, cmap="gray")
    ax[1].set_title('mine')
    plt.show()


def check_zero_cross(img: np.ndarray):
    ans1 = ex2_utils.edgeDetectionZeroCrossingSimple(img)
    ans2 = ex2_utils.edgeDetectionZeroCrossingLOG(img)
    f, ax = plt.subplots(1, 2)
    f.suptitle('Zero Crossing', fontsize=16)
    ax[0].imshow(ans1, cmap="gray")
    ax[0].set_title('simple')
    ax[1].imshow(ans2, cmap="gray")
    ax[1].set_title('LOG')
    plt.show()


def check_canny(img: np.ndarray):
    cvc, myc = ex2_utils.edgeDetectionCanny(img, 100, 50)
    f, ax = plt.subplots(1, 2)
    f.suptitle('canny', fontsize=16)
    ax[0].imshow(cvc, cmap="gray")
    ax[0].set_title('cv')
    ax[1].imshow(myc, cmap="gray")
    ax[1].set_title('mine')
    plt.show()


def check_hough(img: np.ndarray):
    list = ex2_utils.houghCircle(img, 40, 100)
    f, ax = plt.subplots()
    f.suptitle('hough', fontsize=16)
    ax.imshow(img, cmap="gray")
    for c in list:
        circle1 = plt.Circle((c[0], c[1]), c[2], color='r', fill=False)
        ax.add_artist(circle1)
    plt.show()


def main():
    boxman = cv.imread("boxman.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    coin = cv.imread("coincut.png", cv.IMREAD_GRAYSCALE).astype(np.float32)
    monkey = cv.imread("codeMonkey.jpeg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    frog = cv.imread("frog.png", cv.IMREAD_GRAYSCALE).astype(np.float32)
    check_conv1()
    check_conv2(boxman)
    check_Derivative(frog)
    check_blur(boxman)
    check_sobel(boxman)
    check_zero_cross(monkey)
    check_canny(boxman)
    check_hough(coin)


if __name__ == '__main__':
    main()
