from PIL import Image, ImageFile
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES=True
Image.MAX_IMAGE_PIXELS=None


def img_compose(image_names, H_NUM=None, W_NUM=None):
    if H_NUM is not None and W_NUM is not None:
        H_NUM = (0, H_NUM) if isinstance(H_NUM, int) else H_NUM
        W_NUM = (0, W_NUM) if isinstance(W_NUM, int) else W_NUM
        img_path_arr = np.array([[image_names.format(h, w) for w in range(*W_NUM)] for h in range(*H_NUM)])
        target_img_path = image_names.format("{}~{}".format(*H_NUM), "{}~{}".format(*W_NUM))
    elif H_NUM is None and W_NUM is not None:
        W_NUM = (0, W_NUM) if isinstance(W_NUM, int) else W_NUM
        img_path_arr = np.array([[image_names.format(w) for w in range(*W_NUM)]])
        target_img_path = image_names.format("{}~{}".format(*W_NUM))
    elif H_NUM is not None and W_NUM is None:
        H_NUM = (0, H_NUM) if isinstance(H_NUM, int) else H_NUM
        img_path_arr =np.array([[image_names.format(h)] for h in range(*H_NUM)])
        target_img_path = image_names.format("{}~{}".format(*H_NUM))
    else:
        img_path_arr = np.array(image_names)
        target_img_path = os.path.join(os.path.dirname(image_names[0][0]), "compose_{}x{}{}".format(*img_path_arr.shape[:2], os.path.splitext(image_names[0][0])[1]))
    print(">>>>>>>>>>>>>> target cube image list: \n", img_path_arr)
    print(">>>>>>>>>>>>>> compose image to: \n", target_img_path)
    # exit()

    H_NUM, W_NUM = img_path_arr.shape

    big_image_data = []
    cube_image_shape = None
    for h in range(H_NUM):
        for w in range(W_NUM):
            img_path = img_path_arr[h][w]
            # img_data = np.array(Image.open(img_path).convert("L"))
            img_data = np.array(Image.open(img_path))
            print(">>>>>", img_data.shape)
            cube_image_shape = img_data.shape
            big_image_data.append(img_data)
    big_image_data = np.array(big_image_data)
    big_image_data = big_image_data.reshape(H_NUM, W_NUM, *cube_image_shape)
    print("=========big image data shape:", big_image_data.shape)

    big_image_H = H_NUM * cube_image_shape[0]
    big_image_W = W_NUM * cube_image_shape[1]
    to_image = Image.new("RGB", (big_image_W, big_image_H))
    for h in range(H_NUM):
        h_s = h * cube_image_shape[0]
        for w in range(W_NUM):
            w_s = w * cube_image_shape[1]
            v = big_image_data[h][w]
            to_image.paste(Image.fromarray(v), (w_s, h_s))
    print(np.array(to_image).shape)
    to_image.save(target_img_path)



if __name__ == '__main__':

    base_dir = "D:\\cj\whu_16384_16384_lasted_realmask_aug\\pictmp_random_ds_colorajust2_pic0\\"

    paths = [
        os.path.join(base_dir, "0_class_1_rank{}.png"),
        os.path.join(base_dir, "0_class_2_rank{}.png"),
        os.path.join(base_dir, "0_class_4_rank{}.png"),
        os.path.join(base_dir, "0_class_5_rank{}.png"),
    ]
    image_path = [[
        os.path.join(base_dir, "0_class_1_rank0~32.png"),
        os.path.join(base_dir, "0_class_2_rank0~32.png"),
    ], [
        os.path.join(base_dir, "0_class_4_rank0~32.png"),
        os.path.join(base_dir, "0_class_5_rank0~32.png"),
    ],
    ]

    # base_dir = "D:\\cj\whu_16384_16384_lasted_realmask_aug\\random_ds_pic1\\"

    # paths = [
    #     os.path.join(base_dir, "1_class_1_rank{}.png"),
    #     os.path.join(base_dir, "1_class_2_rank{}.png"),
    #     os.path.join(base_dir, "1_class_4_rank{}.png"),
    #     os.path.join(base_dir, "1_class_5_rank{}.png"),
    # ]
    # image_path = [[
    #     os.path.join(base_dir, "1_class_1_rank0~32.png"),
    #     os.path.join(base_dir, "1_class_2_rank0~32.png"),
    # ], [
    #     os.path.join(base_dir, "1_class_4_rank0~32.png"),
    #     os.path.join(base_dir, "1_class_5_rank0~32.png"),
    # ],
    # ]

    # base_dir = "D:\\cj\whu_16384_16384_lasted_realmask_aug\\random_ds_pic2\\"

    # paths = [
    #     os.path.join(base_dir, "2_class_1_rank{}.png"),
    #     os.path.join(base_dir, "2_class_2_rank{}.png"),
    #     os.path.join(base_dir, "2_class_3_rank{}.png"),
    #     os.path.join(base_dir, "2_class_4_rank{}.png"),
    # ]
    # image_path = [[
    #     os.path.join(base_dir, "2_class_1_rank0~32.png"),
    #     os.path.join(base_dir, "2_class_2_rank0~32.png"),
    # ], [
    #     os.path.join(base_dir, "2_class_3_rank0~32.png"),
    #     os.path.join(base_dir, "2_class_4_rank0~32.png"),
    # ],
    # ]

    base_dir = "D:\\cj\whu_16384_16384_lasted_realmask_aug\\random_ds_pic3\\"

    paths = [
        os.path.join(base_dir, "3_class_1_rank{}.png"),
        os.path.join(base_dir, "3_class_2_rank{}.png"),
        os.path.join(base_dir, "3_class_3_rank{}.png"),
        os.path.join(base_dir, "3_class_4_rank{}.png"),
        os.path.join(base_dir, "3_class_5_rank{}.png"),
        os.path.join(base_dir, "3_class_6_rank{}.png"),
    ]
    image_path = [[
        os.path.join(base_dir, "3_class_1_rank0~32.png"),
        os.path.join(base_dir, "3_class_2_rank0~32.png"),
    ], [
        os.path.join(base_dir, "3_class_3_rank0~32.png"),
        os.path.join(base_dir, "3_class_4_rank0~32.png"),
    ], [
        os.path.join(base_dir, "3_class_5_rank0~32.png"),
        os.path.join(base_dir, "3_class_6_rank0~32.png"),
    ],
    ]

    # for path in paths:
    #     img_compose(path, W_NUM=32)
    img_compose(image_path)
