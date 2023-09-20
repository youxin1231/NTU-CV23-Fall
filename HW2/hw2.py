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

def plot_histogram(img):
    count = np.zeros(256, dtype=int)
    height, width = img.shape
    for h in range(height):
        for w in range(width):
            count[img[h, w]] += 1
    plt.bar(range(256), count)
    plt.xlabel('Grayscale intensity')
    plt.ylabel('Number of pixels')
    plt.savefig('b.png')

def plot_connected_components(img):
    # The iterative algorithm
    height, width = img.shape
    label = np.zeros((height + 2, width + 2), int)
    
    # Init
    tmp = 1
    for h in range(1, height+1):
        for w in range(1, width+1):
            if(img[h-1, w-1] == 255):
                label[h, w] = tmp
                tmp += 1
    # Iterative
    while(1):
        # Top-down pass
        change = False

        for h in range(1, height+1):
            for w in range(1, width+1):
                if(label[h, w] != 0):
                    # 4-connectivity
                    neighbors = [label[h-1, w], label[h+1,w], label[h, w-1], label[h, w+1], label[h, w]]
                    M = min([x for x in neighbors if x != 0])
                    if (M != label[h, w]):
                        change = True
                        label[h, w] = M

        # Bottom-up pass

        for h in reversed(range(1, height+1)):
            for w in reversed(range(1, width+1)):
                if(label[h, w] != 0):
                    # 4-connectivity
                    neighbors = [label[h-1, w], label[h+1,w], label[h, w-1], label[h, w+1], label[h, w]]
                    M = min([x for x in neighbors if x != 0])
                    if (M != label[h, w]):
                        change = True
                        label[h, w] = M

        if (not change):
            break
    label = label[1:-1, 1:-1]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for uni_label in np.unique(label):
        if uni_label == 0:
            continue
        
        mask = np.where(label == uni_label, 1, 0)
        if np.sum(mask) < 500:
            continue
        
        mask = np.argwhere(mask != 0)
        min_h, min_w = np.min(mask, axis=0)
        max_h, max_w = np.max(mask, axis=0)
        
        centroid_h = (min_h + max_h) // 2
        centroid_w = (min_w + max_w) // 2

        img = cv2.rectangle(img, (min_w, min_h), (max_w, max_h), (255, 0, 0), 5)
        img = cv2.drawMarker(img, (centroid_w, centroid_h), color = (0, 0, 255), markerType = cv2.MARKER_CROSS, thickness = 2)
        
    cv2.imwrite('c.png', img)


def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)

    # (a) a binary image (threshold at 128)
    binary_img = binarize(img)
    cv2.imwrite('a.png', binary_img)

    # (b) a histogram
    plot_histogram(img)

    # (c) connected components (regions with + at centroid, bounding box)
    plot_connected_components(binary_img)

if __name__ == '__main__':
    main()