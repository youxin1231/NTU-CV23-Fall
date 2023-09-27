import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(img, hist_name):
    count = np.zeros(256, dtype=int)
    height, width = img.shape
    for h in range(height):
        for w in range(width):
            count[img[h, w]] += 1
    plt.bar(range(256), count)
    plt.xlabel('Grayscale intensity')
    plt.ylabel('Number of pixels')
    plt.savefig(hist_name)
    plt.clf()

def intensity_divided_by_three(img):
    height, width = img.shape
    new_img = np.zeros_like(img)
    
    for h in range(height):
        for w in range(width):
            new_img[h, w] = img[h, w] // 3

    return new_img

def histogram_equalization(img):
    map = np.zeros(256, dtype=float)
    count = np.zeros(256, dtype=int)

    height, width = img.shape
    total_pixels = height * width
    for h in range(height):
        for w in range(width):
            count[img[h, w]] += 1
    
    for k in range(256):
        for j in range(k):
            map[k] += 255 * count[j] / total_pixels

    new_img = np.zeros_like(img)
    for h in range(height):
        for w in range(width):
            new_img[h, w] = int(map[img[h, w]])
    
    return new_img

def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)

    # (a) original image and its histogram
    cv2.imwrite('a_img.png', img)
    plot_histogram(img, 'a_hist.png')

    # (b) image with intensity divided by 3 and its histogram
    b_img = intensity_divided_by_thress(img)
    cv2.imwrite('b_img.png', b_img)
    plot_histogram(b_img, 'b_hist.png')

    # (c) image after applying histogram equalization to (b) and its histogram
    c_img = histogram_equalization(b_img)
    cv2.imwrite('c_img.png', c_img)
    plot_histogram(c_img, 'c_hist.png')

if __name__ == '__main__':
    main()