import cv2
import numpy as np

def upside_down(img):
    height, width, _ = img.shape
    upside_down_img = np.zeros_like(img)

    for h in range(height):
        upside_down_img[height-1-h] = img[h]
    
    return upside_down_img

def right_side_left(img):
    height, width, _ = img.shape
    right_side_left_img = np.zeros_like(img)

    for w in range(width):
        right_side_left_img[:, width-1-w] = img[:, w]

    return right_side_left_img

def diagonally_flip(img):
    height, width, _ = img.shape
    diagonally_flip_img = np.zeros_like(img)

    for h in range(height):
        for w in range(width):
            diagonally_flip_img[w, h] = img[h, w]

    return diagonally_flip_img

def rotate_degree(img, degree):
    height, width, _ = img.shape
    center = (height // 2, width // 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    
    rotate_img = cv2.warpAffine(img, M, (width, height))
    return rotate_img

def shrink(img, ratio):
    height, width, _ = img.shape
    new_height = height // ratio
    new_width = width // ratio
    shrink_img = np.zeros((new_height, new_width, 3))

    for h in range(new_height):
        for w in range(new_width):
            shrink_img[h, w] = img[h * ratio, w * ratio]
    
    return shrink_img
    
def binarize(img):
    height, width, _ = img.shape
    binarize_img = np.zeros_like(img)
    
    for h in range(height):
        for w in range(width):
            if((img[h, w] >= 128).any()):
                binarize_img[h, w] = 255
            else:
                binarize_img[h, w] = 0
    return binarize_img

def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path)

    # Part1. Write a program to do the following requirement.
    # (a) upside-down lena.bmp
    cv2.imwrite('a.bmp', upside_down(img))

    # (b) right-side-left lena.bmp
    cv2.imwrite('b.bmp', right_side_left(img))

    # (c) diagonally flip lena.bmp
    cv2.imwrite('c.bmp', diagonally_flip(img))

    # Part2. Write a program or use software to do the following requirement.
    # (d) rotate lena.bmp 45 degrees clockwise
    cv2.imwrite('d.bmp', rotate_degree(img, -45))

    # (e) shrink lena.bmp in half
    cv2.imwrite('e.bmp', shrink(img, 2))

    # (f) binarize lena.bmp at 128 to get a binary image
    cv2.imwrite('f.bmp', binarize(img))

if __name__ == '__main__':
    main()