import cv2
import numpy as np

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img, s):
    k = int(s) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), s)

def edges(img):
    return cv2.Canny(img, 50, 150)

def diffract(img, s):
    g = to_gray(img)
    b = blur(g, s)
    out = np.zeros_like(b)
    out[b > 50] = b[b > 50]
    return out