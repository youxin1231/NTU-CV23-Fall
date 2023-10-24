import cv2
import numpy as np

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

def shrink(img, ratio):
    height, width = img.shape
    new_height = height // ratio
    new_width = width // ratio
    shrink_img = np.zeros((new_height, new_width))

    for h in range(new_height):
        for w in range(new_width):
            shrink_img[h, w] = img[h * ratio, w * ratio]
    
    return shrink_img

def yokoi_connectivity(img):
    height, width = img.shape

    pad_img = np.zeros((height+2, width+2))
    pad_img[1:height+1, 1:width+1] = img

    output_img = np.zeros_like(img) - 1

    for h in range(1, height+1):
        for w in range(1, width+1):
            if pad_img[h, w] == 255:
                x7, x2, x6 = pad_img[h-1, w-1], pad_img[h-1, w], pad_img[h-1, w+1]
                x3, x0, x1 = pad_img[h, w-1], pad_img[h, w], pad_img[h, w+1]
                x8, x4, x5 = pad_img[h+1, w-1], pad_img[h+1, w], pad_img[h+1, w+1]
                
                a1 = yokoi_h_function(x0, x1, x6, x2)
                a2 = yokoi_h_function(x0, x2, x7, x3)
                a3 = yokoi_h_function(x0, x3, x8, x4)
                a4 = yokoi_h_function(x0, x4, x5, x1)

                output_img[h-1, w-1] = yokoi_f_function(a1, a2, a3, a4)

    return output_img

def yokoi_h_function(b, c, d, e):
    if b == c and (b != d or b != e):
        return 'q'
    elif b == c == d == e:
        return 'r'
    else:
        return 's'
        
def yokoi_f_function(a1, a2, a3, a4):
    if a1 == a2 == a3 == a4 == 'r':
        return 5
    else:
        return [a1, a2, a3, a4].count('q')
    
def pair_relationship(img):
    height, width = img.shape

    pad_img = np.zeros((height+2, width+2))
    pad_img[1:height+1, 1:width+1] = img

    output_img = np.zeros((height, width))
    
    for h in range(1, height+1):
        for w in range(1, width+1):
            x7, x2, x6 = pad_img[h-1, w-1], pad_img[h-1, w], pad_img[h-1, w+1]
            x3, x0, x1 = pad_img[h, w-1], pad_img[h, w], pad_img[h, w+1]
            x8, x4, x5 = pad_img[h+1, w-1], pad_img[h+1, w], pad_img[h+1, w+1]
            if x0 == -1:
                output_img[h-1, w-1] = 0
            else:
                output_img[h-1, w-1] = pair_f_function(x0, x1, x2, x3, x4)

    return output_img

def pair_h_function(a, m):
    if a == m:
        return 1
    else:
        return 0
    
def pair_f_function(x0, x1, x2, x3, x4):
    m = 1
    if (pair_h_function(x1, m) + pair_h_function(x2, m) + pair_h_function(x3, m) + pair_h_function(x4, m)) >= 1 and x0 == m:
        return 1
    else:
        return 2

def connected_shrink(img):
    height, width = img.shape

    tmp_img = np.zeros((height, width))
    tmp_img[img > 0] = 1
    pad_img = np.zeros((height+2, width+2))
    pad_img[1:height+1, 1:width+1] = tmp_img

    for h in range(1, height+1):
        for w in range(1, width+1):
            if img[h-1, w-1] == 1:
                x7, x2, x6 = pad_img[h-1, w-1], pad_img[h-1, w], pad_img[h-1, w+1]
                x3, x0, x1 = pad_img[h, w-1], pad_img[h, w], pad_img[h, w+1]
                x8, x4, x5 = pad_img[h+1, w-1], pad_img[h+1, w], pad_img[h+1, w+1]
                
                a1 = connected_h_function(x0, x1, x6, x2)
                a2 = connected_h_function(x0, x2, x7, x3)
                a3 = connected_h_function(x0, x3, x8, x4)
                a4 = connected_h_function(x0, x4, x5, x1)

                if connected_f_function(a1, a2, a3, a4, 1) == 0:
                    img[h-1, w-1] = 0
                    pad_img[h, w] = 0

            if img[h-1, w-1] != 0:
                img[h-1, w-1] = 255

    return img

def connected_h_function(b, c, d, e):
    if b == c and (b != d or b != e):
        return 1
    else:
        return 0
    
def connected_f_function(a1, a2, a3, a4, x):
    if [a1, a2, a3, a4].count(1) == 1:
        return 0
    else:
        return x
    
def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)

    # Down-sampling
    img = binarize(img)
    img = shrink(img, 8)
    counter = 1
    while True:
        tmp_img = img.copy()
        yokoi_map = yokoi_connectivity(img)
        pair_map = pair_relationship(yokoi_map)
        img = connected_shrink(pair_map)
        if (tmp_img == img).all():
            break

        cv2.imwrite(f'thinning_{counter}.png', img)
        counter += 1

if __name__ == '__main__':
    main()
