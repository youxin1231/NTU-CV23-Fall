import cv2
import numpy as np

def dilation(img, kernel):
    height, width = img.shape
    half_ks = kernel.shape[0] // 2

    output_img = np.zeros_like(img)
    for h in range(half_ks, height-half_ks):
        for w in range(half_ks, width-half_ks):
            region = img[h-half_ks:h+half_ks+1, w-half_ks:w+half_ks+1]

            output_img[h, w] = np.max(region[kernel == 1])

    return output_img

def erosion(img, kernel):
    height, width = img.shape
    half_ks = kernel.shape[0] // 2

    output_img = np.zeros_like(img)
    for h in range(half_ks, height-half_ks):
        for w in range(half_ks, width-half_ks):
            region = img[h-half_ks:h+half_ks+1, w-half_ks:w+half_ks+1]

            output_img[h, w] = np.min(region[kernel == 1])

    return output_img

def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)

    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]], dtype=np.uint8)
    
    # (a) Dilation
    dilation_img = dilation(img, kernel)
    cv2.imwrite('a.png', dilation_img)

    # (b) Erosion
    erosion_img = erosion(img, kernel)
    cv2.imwrite('b.png', erosion_img)

    # (c) Opening
    opening_img = dilation(erosion_img, kernel)
    cv2.imwrite('c.png', opening_img)

    # (d) Closing
    closing_img = erosion(dilation_img, kernel)
    cv2.imwrite('d.png', closing_img)

if __name__ == '__main__':
    main()