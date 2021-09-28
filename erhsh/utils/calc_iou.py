import os
import numpy as np
import cv2
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES=True
Image.MAX_IMAGE_PIXELS=None


def __cal_hist(a, b, n):
    # a -> label_true, b -> label_predict
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)

def calc_hist(label_img, predict_img, num_classes):
    label_np = label_img
    predict_np = predict_img

    if isinstance(label_img, str):
        label_np = np.array(Image.open(label_img).convert('L'), dtype=np.uint8)
    if isinstance(predict_img, str):
        predict_np = np.array(Image.open(predict_img).convert('L'), dtype=np.uint8)

    print(">>>label_np shape:", label_np.shape)
    print(">>>predict_np shape:", predict_np.shape)

    return __cal_hist(label_np.flatten(), predict_np.flatten(), num_classes)


def calc_mean_iou(label_imgs, predict_imgs, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lable_img, predict_img in zip(label_imgs, predict_imgs):
        hist += calc_hist(lable_img, predict_img, num_classes)

    print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_iou = np.nanmean(iu)
    print('per-class IoU', iu)
    print('mean IoU', mean_iou)
    return mean_iou


if __name__ == '__main__':
    names = """
        GF2_PMS1__L1A0000647768-MSS1
        GF2_PMS1__L1A0001094941-MSS1
        GF2_PMS1__L1A0001169100-MSS1
        GF2_PMS1__L1A0001491417-MSS1
        GF2_PMS1__L1A0001537716-MSS1
        GF2_PMS1__L1A0001553848-MSS1
        GF2_PMS1__L1A0001562063-MSS1
        GF2_PMS1__L1A0001734328-MSS1
        GF2_PMS2__L1A0000564692-MSS2
        GF2_PMS2__L1A0000647892-MSS2
        GF2_PMS2__L1A0000718813-MSS2
        GF2_PMS2__L1A0000948183-MSS2
        GF2_PMS2__L1A0001015596-MSS2
        GF2_PMS2__L1A0001028977-MSS2
        GF2_PMS2__L1A0001035795-MSS2
        GF2_PMS2__L1A0001119060-MSS2
        GF2_PMS2__L1A0001336975-MSS2
        GF2_PMS2__L1A0001378491-MSS2
        GF2_PMS2__L1A0001389317-MSS2
        GF2_PMS2__L1A0001416140-MSS2
        GF2_PMS2__L1A0001457157-MSS2
        GF2_PMS2__L1A0001537637-MSS2
        GF2_PMS2__L1A0001566653-MSS2
        GF2_PMS2__L1A0001577567-MSS2
        GF2_PMS2__L1A0001633212-MSS2
        GF2_PMS2__L1A0001708232-MSS2
        GF2_PMS2__L1A0001757317-MSS2
        GF2_PMS2__L1A0001757484-MSS2
        GF2_PMS2__L1A0001787080-MSS2
        GF2_PMS2__L1A0001886305-MSS2
    """

    names = names.split()
    base_dir = "/home/cj/pictmp.gid_gray_comp/"
    num_classes = 6

    lbl_paths = [os.path.join(base_dir, "{}_lbl_rank{}.tif".format(x, "0~32")) for x in names]
    img_paths = [os.path.join(base_dir, "{}_rank{}.tif".format(x, "0~32")) for x in names]

    calc_mean_iou(lbl_paths, img_paths, num_classes)