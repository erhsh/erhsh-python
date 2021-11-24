import argparse

def sync_data(from_path, to_path):
    print("from path:", from_path)
    print("to path:", to_path)
    import moxing as mox
    mox.file.copy_parallel(from_path, to_path)
    print("===finish data synch===")

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, help="dataset url")
parser.add_argument("--train_url", type=str, help="output url")
parser.add_argument("--patch_url", type=str, help="patches url")

args = parser.parse_args()
print(">>>>>>>>args:", args)

from_path = args.patch_url
to_path = "/cache/data/"
sync_data(from_path, to_path)

# import os
# 
# os.system("pip install /cache/data/*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple")
# os.system("ifconfig endvnic 192.168.1.200/24")
# print(">>>>>>>>> ifconfig success.")
# os.system("sshpass -p 'Huawei2012#' scp /cache/data/libcpu_kernels_GridSampler3d.SO HwHiAiUser@192.168.1.199:/usr/lib64/aicpu_kernels/")
# print(">>>>>>>>> scp 199 success.")
# os.system("sshpass -p 'Huawei2012#' scp /cache/data/libcpu_kernels_GridSampler3d.SO HwHiAiUser@192.168.1.195:/usr/lib64/aicpu_kernels/")
# print(">>>>>>>>> scp 195 success.")