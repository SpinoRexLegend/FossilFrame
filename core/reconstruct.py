import numpy as np
import cv2

def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

def fuse(blurred, phase, alpha):
    b = blurred.astype(np.float32) / 255.0
    p = phase.astype(np.float32)
    if p.max() > 0:
        p = p / p.max()
    out = b + alpha * p
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

def enhance(blurred, phase, alpha):
    fused = fuse(blurred, phase, alpha)
    sharp = sharpen(fused)
    out = np.zeros_like(sharp)
    out[blurred > 50] = sharp[blurred > 50]
    return out