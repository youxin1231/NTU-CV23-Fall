import cv2
import numpy as np

def Robert_operator(img, thres):
    pad_pixel = 1
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    R1 = np.array([
        [-1, 0],
        [0, 1]], dtype=np.int8)
    
    R2 = np.array([
        [0, -1],
        [1, 0]], dtype=np.int8)
    
    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h:h + 2, w:w + 2]

            r1 = np.sum(patch * R1)
            r2 = np.sum(patch * R2)

            grad = np.hypot(r1, r2)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def Prewitt_detector(img, thres):
    pad_pixel = 1
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    P1 = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]], dtype=np.int8)
    
    P2 = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]], dtype=np.int8)
    
    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h - 1:h + 2, w - 1:w + 2]

            p1 = np.sum(patch * P1)
            p2 = np.sum(patch * P2)

            grad = np.hypot(p1, p2)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def Sobel_detector(img, thres):
    pad_pixel = 1
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    S1 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]], dtype=np.int8)
    
    S2 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.int8)
    
    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h - 1:h + 2, w - 1:w + 2]

            s1 = np.sum(patch * S1)
            s2 = np.sum(patch * S2)

            grad = np.hypot(s1, s2)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def Frei_N_Chen_operator(img, thres):
    pad_pixel = 1
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    F1 = np.array([
        [-1, -np.sqrt(2), -1],
        [0, 0, 0],
        [1, np.sqrt(2), 1]])
    
    F2 = np.array([
        [-1, 0, 1],
        [-np.sqrt(2), 0, np.sqrt(2)],
        [-1, 0, 1]])
    
    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h - 1:h + 2, w - 1:w + 2]

            f1 = np.sum(patch * F1)
            f2 = np.sum(patch * F2)

            grad = np.hypot(f1, f2)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def Kirsch_operator(img, thres):
    pad_pixel = 1
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    Ks = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]
    
    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h - 1:h + 2, w - 1:w + 2]

            ks = [np.sum(patch * K) for K in Ks]
            grad = np.max(ks)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def Robinson_operator(img, thres):
    pad_pixel = 1
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    Rs = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])]
    
    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h - 1:h + 2, w - 1:w + 2]

            rs = [np.sum(patch * R) for R in Rs]
            grad = np.max(rs)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def Nevatia_Babu_operator(img, thres):
    pad_pixel = 2
    padding_img = cv2.copyMakeBorder(img, pad_pixel, pad_pixel, pad_pixel, pad_pixel, cv2.BORDER_REFLECT)

    height, width = padding_img.shape
    output_img = np.zeros_like(img)
    
    Ns = [
        np.array([[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]]),
        np.array([[100, 100, 100, 100, 100], [100, 100, 100, 78, -32], [100, 92, 0, -92, -100], [32, -78, -100, -100, -100], [-100, -100, -100, -100, -100]]),
        np.array([[100, 100, 100, 32, -100], [100, 100, 92, -78, -100], [100, 100, 0, -100, -100], [100, 78, -92, -100, -100], [100, -32, -100, -100, -100]]),
        np.array([[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100]]),
        np.array([[-100, 32, 100, 100, 100], [-100, -78, 92, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, -92, 78, 100], [-100, -100, -100, -32, 100]]),
        np.array([[100, 100, 100, 100, 100], [-32, 78, 100, 100, 100], [-100, -92, 0, 92, 100], [-100, -100, -100, -78, 32], [-100, -100, -100, -100, -100]])]

    for h in range(pad_pixel, height - pad_pixel):
        for w in range(pad_pixel, width - pad_pixel):
            patch = padding_img[h - 2:h + 3, w - 2:w + 3]

            ns = [np.sum(patch * N) for N in Ns]
            grad = np.max(ns)
            if grad > thres:
                output_img[h - pad_pixel, w - pad_pixel] = 0
            else:
                output_img[h - pad_pixel, w - pad_pixel] = 255
    
    return output_img

def main():
    img_path = 'lena.bmp'
    img = cv2.imread(img_path, 0)

    # (a) Robert's Operator: 12
    cv2.imwrite('a.png', Robert_operator(img, 12))
    
    # (b) Prewitt's Edge Detector: 24
    cv2.imwrite('b.png', Prewitt_detector(img, 24))

    # (c) Sobel's Edge Detector: 38
    cv2.imwrite('c.png', Sobel_detector(img, 38))

    # (d) Frei and Chen's Gradient Operator: 30
    cv2.imwrite('d.png', Frei_N_Chen_operator(img, 30))

    # (e) Kirsch's Compass Operator: 135
    cv2.imwrite('e.png', Kirsch_operator(img, 135))

    # (f) Robinson's Compass Operator: 43
    cv2.imwrite('f.png', Robinson_operator(img, 43))

    # (g) Nevatia-Babu 5x5 Operator: 12500
    cv2.imwrite('g.png', Nevatia_Babu_operator(img, 12500))
    
if __name__ == '__main__':
    main()