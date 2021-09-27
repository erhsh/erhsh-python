import os


template_txt = """
-- base dir
/home/cj/workspace

-- image matrix (2, 2, 4)
img_0_0.png img_0_1.png img_0_2.png img_0_3.png, img_1_0.png img_1_1.png img_1_2.png img_1_3.png 
img_2_0.png img_2_1.png img_2_2.png img_2_3.png, img_3_0.png img_3_1.png img_3_2.png img_3_3.png 

-- output image path
/home/cj/workspace/compose_img.png
"""


def gen_template():
    print(template_txt)


if __name__ == '__main__':
    gen_template()