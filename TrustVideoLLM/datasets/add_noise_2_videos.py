import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from PIL import ImageEnhance, ImageFilter, ImageDraw



def add_blur(frame):
    # 随机选择模糊程度
    kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def add_occlusion(frame):
    # 随机选择遮挡区域
    h, w, _ = frame.shape
    x1 = random.randint(0, w // 2)
    y1 = random.randint(0, h // 2)
    x2 = random.randint(x1, w)
    y2 = random.randint(y1, h)
    frame[y1:y2, x1:x2] = 0  # 用黑色矩形遮挡
    return frame

def add_color_noise(frame):
    # 随机添加颜色噪声
    noise = np.random.randint(-50, 50, frame.shape, dtype=np.int32)
    frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return frame
