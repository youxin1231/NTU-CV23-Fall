import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize(img):
    height, width = img.shape
    binarize_img = np.zeros_like(img)
    
    for h in range(height):
        for w in range(width):
            if((img[h, w] >= 128)):
                binarize_img[h, w] = 255
            else:
                binarize_img[h, w] = 0
    return binarize_img

def dilation(img, kernel):
    height, width = img.shape
    half_ks = kernel.shape[0] // 2

    output_img = np.zeros_like(img)
    for h in range(half_ks, height-half_ks):
        for w in range(half_ks, width-half_ks):
            region = img[h-half_ks:h+half_ks+1, w-half_ks:w+half_ks+1]

            if np.sum(region * kernel) > 0:
                output_img[h-half_ks, w-half_ks] = 255

    return output_img

def erosion(img, kernel):
    height, width = img.shape
    half_ks = kernel.shape[0] // 2

    output_img = np.zeros_like(img)
    for h in range(half_ks, height-half_ks):
        for w in range(half_ks, width-half_ks):
            region = img[h-half_ks:h+half_ks+1, w-half_ks:w+half_ks+1]

            if np.all(region[kernel == 1] == 255):
                output_img[h-half_ks, w-half_ks] = 255

    return output_img

def hit_and_miss(img, J_kernel, K_kernel):
    height, width = img.shape
    half_ks = max(J_kernel.shape[0] // 2, K_kernel.shape[0] // 2)
    
    output_img = np.zeros_like(img)

    for h in range(half_ks, height-half_ks):
        for w in range(half_ks, width-half_ks):
            region = img[h-half_ks:h+half_ks+1, w-half_ks:w+half_ks+1]
        
            if np.all(region[J_kernel == 1] == 255) and np.all(~region[K_kernel == 1] == 255):
                    output_img[h, w] = 255

    return output_img

def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)
    binary_img = binarize(img)

    octogonal_kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]], dtype=np.uint8)
    
    # (a) Dilation
    dilation_img = dilation(binary_img, octogonal_kernel)
    cv2.imwrite('a.png', dilation_img)

    # (b) Erosion
    erosion_img = erosion(binary_img, octogonal_kernel)
    cv2.imwrite('b.png', erosion_img)

    # (c) Opening
    opening_img = dilation(erosion_img, octogonal_kernel)
    cv2.imwrite('c.png', opening_img)

    # (d) Closing
    closing_img = erosion(dilation_img, octogonal_kernel)
    cv2.imwrite('d.png', closing_img)

    # (e) Hit-and-miss transform
    J_kernel = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0]], dtype=np.uint8)
    
    K_kernel = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]], dtype=np.uint8)

    hnm_img = hit_and_miss(binary_img, J_kernel, K_kernel)
    cv2.imwrite('e.png', hnm_img)

if __name__ == '__main__':
    main()