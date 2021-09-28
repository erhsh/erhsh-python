from PIL import Image, ImageFile
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES=True
Image.MAX_IMAGE_PIXELS=None

PIXEL_TO_RGB = {
    0: [0, 0, 0],   # blank
    1: [255, 0, 0], # red           --> bare-land 
    2: [0, 255, 0], # green         
    3: [0, 0, 255], # blue          --> vegetation
    4: [255, 255, 0], # yellow
    5: [255, 0, 255], # pink        --> building
    6: [0, 255, 255], # cyan        --> road
    7: [127, 0, 0], # dark red
    8: [0, 127, 0], # dark green
    9: [0, 0, 127], # dark blue
    10: [127, 127, 0], # dark yellow --> warter-body
    11: [127, 0, 127], # dark pink
    12: [0, 127, 127], # dark cyan
}

RGB_TO_PIXEL = {"_".join([str(v) for v in value]): key for key, value in PIXEL_TO_RGB.items()}


def gray2RGB(img):
    H, W = img.shape
    print(">>>>>>>>>>", img)
    img_rgb = np.random.randint(0, 256, size=[H, W, 3], dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            k = img[i][j]
            rgb = PIXEL_TO_RGB.get(k, [255, 255, 255])
            img_rgb[i][j] = rgb
    return img_rgb


def RGB2gray(img_rgb):
    H, W, C = img_rgb.shape
    img = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            rgb = img_rgb[i][j]
            rgb = "_".join([str(v) for v in rgb])
            k = RGB_TO_PIXEL.get(rgb, 255)
            img[i][j] = k
    return img


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


def convertGray2RGB_Muti(img_path, dest_path=None, h_num=1, w_num=1):
    print(">>> img_path={}, dest_path={}".format(img_path, dest_path))
    img_data = Image.open(img_path).convert("L")
    img_np = np.array(img_data)
 
    from erhsh import utils as eut

    processor = eut.MutiProcessor(img_np, h_num, w_num, cube_func=gray2RGB)
    ret = processor.process()

    if dest_path is None:
        name = os.path.splitext(img_path)[0]
        ext = os.path.splitext(img_path)[1]
        dest_path = "{}_rgb{}".format(name, ext)
    elif os.path.isdir(dest_path):
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]
        ext = os.path.splitext(basename)[1]
        dest_path = os.path.join(dest_path, "{}_rgb{}".format(name, ext))

    to_image = Image.new("RGB", img_np.shape[:2][::-1])
    for k, v in ret.items():
        h_s, w_s, _, _ = tuple([int(x) for x in k.split("_")])
        to_image.paste(Image.fromarray(v), (w_s, h_s))

    to_image.save(dest_path)


def convertRGB2Gray(img_path, dest_path=None):
    print(">>> img_path={}, dest_path={}".format(img_path, dest_path))
    img_data = Image.open(img_path)
    img_rgb = np.array(img_data)
 
    print(">>> RGB2gray before, img_rgb.shape={}".format(img_rgb.shape))
    img_gray = RGB2gray(img_rgb)
    print(">>> RGB2gray after, img_gray.shape={}".format(img_gray.shape))

    img_gray = Image.fromarray(img_gray)

    if dest_path is None:
        name = os.path.splitext(img_path)[0]
        ext = os.path.splitext(img_path)[1]
        dest_path = "{}_gray{}".format(name, ext)
    elif os.path.isdir(dest_path):
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]
        ext = os.path.splitext(basename)[1]
        dest_path = os.path.join(dest_path, "{}_gray{}".format(name, ext))

    print(">>> save to dest_path={} start...".format(dest_path))
    img_gray.save(dest_path)
    print(">>> save to dest_path={} success.".format(dest_path))


def convertRGB2Gray_Muti(img_path, dest_path=None, h_num=1, w_num=1):
    print(">>> img_path={}, dest_path={}".format(img_path, dest_path))
    img_data = Image.open(img_path)
    img_rgb = np.array(img_data)
 
    from erhsh import utils as eut

    processor = eut.MutiProcessor(img_rgb, h_num, w_num, cube_func=RGB2gray)
    ret = processor.process()

    if dest_path is None:
        name = os.path.splitext(img_path)[0]
        ext = os.path.splitext(img_path)[1]
        dest_path = "{}_gray{}".format(name, ext)
    elif os.path.isdir(dest_path):
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]
        ext = os.path.splitext(basename)[1]
        dest_path = os.path.join(dest_path, "{}_gray{}".format(name, ext))

    to_image = Image.new("L", img_rgb.shape[:2][::-1])
    for k, v in ret.items():
        h_s, w_s, _, _ = tuple([int(x) for x in k.split("_")])
        to_image.paste(Image.fromarray(v), (w_s, h_s))

    to_image.save(dest_path)


if __name__ == '__main__':
    mock_gray_img_path = "tmp_gray.png"
    mock_rgb_img_path = "tmp_rgb.png"
    mock_gray_img_np = np.random.randint(1, 10, (100, 100))
    mock_gray_img = Image.fromarray(mock_gray_img_np)
    mock_gray_img.save(mock_gray_img_path)
    convertGray2RGB(mock_gray_img_path, dest_path=mock_rgb_img_path)
    convertRGB2Gray(mock_rgb_img_path, dest_path=mock_gray_img_path)
    mock_gray_img_np2 = np.array(Image.open(mock_gray_img_path).convert("L"))
    assert np.all(mock_gray_img_np == mock_gray_img_np2)
    os.remove(mock_gray_img_path)
    os.remove(mock_rgb_img_path)