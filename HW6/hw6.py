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

    pad_img = np.zeros((height+2, width+2)) - 1
    pad_img[1:height+1, 1:width+1] = img

    output_img = np.zeros_like(img) - 1

    for h in range(1, height+1):
        for w in range(1, width+1):
            if pad_img[h, w] == 255:
                a1 = h_function(pad_img[h, w], pad_img[h, w+1], pad_img[h-1, w+1], pad_img[h-1, w])
                a2 = h_function(pad_img[h, w], pad_img[h-1, w], pad_img[h-1, w-1], pad_img[h, w-1])
                a3 = h_function(pad_img[h, w], pad_img[h, w-1], pad_img[h+1, w-1], pad_img[h+1, w])
                a4 = h_function(pad_img[h, w], pad_img[h+1, w], pad_img[h+1, w+1], pad_img[h, w+1])

                output_img[h-1, w-1] = f_function(a1, a2, a3, a4)

    return output_img

def h_function(b, c, d, e):
    if b == c and (b != d or b != e):
        return 'q'
    elif b == c == d == e:
        return 'r'
    else:
        return 's'
        
def f_function(a1, a2, a3, a4):
    if a1 == a2 == a3 == a4 == 'r':
        return 5
    else:
        return [a1, a2, a3, a4].count('q')
    
def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)

    # Down-sampling
    img = binarize(img)
    img = shrink(img, 8)
    
    # Yokoi connectivity
    yokoi_map = yokoi_connectivity(img)

    height, width = yokoi_map.shape
    with open('yokoi.txt', 'w') as file:
        for h in range(height):
            for w in range(width):
                if yokoi_map[h, w] != -1:
                    file.write(f'{int(yokoi_map[h, w])}')
                else:
                    file.write(f' ')
            file.write('\n')

if __name__ == '__main__':
    main()