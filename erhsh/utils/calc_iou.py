import os
import numpy as np
import cv2
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES=True
Image.MAX_IMAGE_PIXELS=None


def __cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)

def calc_mean_iou(label_img, predict_img):
    label_np = label_img
    predict_np = predict_img

    if isinstance(label_img, str):
        label_np = np.array(Image.open(label_img).convert('L'), dtype=np.uint8)
    if isinstance(predict_img, str):
        predict_np = np.array(Image.open(predict_img).convert('L'), dtype=np.uint8)

    print(">>>label_np shape:", label_np.shape)
    print(">>>predict_np shape:", predict_np.shape)

    hist = __cal_hist(predict_np.flatten(), label_np.flatten(), 11)
    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_iou = np.nanmean(iu)
    print('per-class IoU', iu)
    print('mean IoU', mean_iou)
    return mean_iou


if __name__ == '__main__':
    label_img_path = "D:\\cj\\whu_16384\\my_predict_pic2_bs1_train_with_p2_dynamic_lossscale_lr128\\compose_2x2_gray.png"
    predict_img_path = "D:\\cj\\whu_16384\\train\\Annotations\\compose_2x2_gray.png"
    mean_iou = calc_mean_iou(label_img_path, predict_img_path)