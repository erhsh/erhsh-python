from PIL import Image, ImageFile
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES=True
Image.MAX_IMAGE_PIXELS=None

PIXEL_TO_RGB = {
    0: [0, 0, 0],   # blank
    1: [255, 0, 0], # red
    2: [0, 255, 0], # green
    3: [0, 0, 255], # blue
    4: [255, 255, 0],
    5: [255, 0, 255],
    6: [0, 255, 255],
    7: [127, 0, 0], 
    8: [0, 127, 0],
    9: [0, 0, 127],
    10: [127, 127, 0],
    11: [127, 0, 127],
    12: [0, 127, 127]
}

def gray2RGB(img):
    H, W = img.shape
    img_rgb = np.random.randint(0, 256, size=[H, W, 3], dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            k = img[i][j]
            rgb = PIXEL_TO_RGB.get(k, [255, 255, 255])
            img_rgb[i][j] = rgb
    return img_rgb


def convertGray2RGB(img_path, dest_path=None):
    print(">>> img_path={}, dest_path={}".format(img_path, dest_path))
    img_data = Image.open(img_path).convert("L")
    img_np = np.array(img_data)
 
    print(">>> gray2RGB before, img_np.shape={}".format(img_np.shape))
    img_np = gray2RGB(img_np)
    print(">>> gray2RGB after, img_np.shape={}".format(img_np.shape))

    im = Image.fromarray(img_np)

    if dest_path is None:
        name = os.path.splitext(img_path)[0]
        ext = os.path.splitext(img_path)[1]
        dest_path = "{}_rgb{}".format(name, ext)
    elif os.path.isdir(dest_path):
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]
        ext = os.path.splitext(basename)[1]
        dest_path = os.path.join(dest_path, "{}_rgb{}".format(name, ext))

    print(">>> save to dest_path={} start...".format(dest_path))
    im.save(dest_path)
    print(">>> save to dest_path={} success.".format(dest_path))

if __name__ == '__main__':
    rand_np = np.random.randint(0, 11, (512, 512), dtype=np.uint8)
    img_np = Image.fromarray(rand_np)
    img_np.save("tmp.png")
    convertGray2RGB("tmp.png", "./")