import argparse
import os

import numpy as np

TYPE_MAPPER = {
    "Float32": np.float32,
    "Float16": np.float16,
    "Uint8": np.uint8,
}


class BinLoader:
    def __init__(self, bin_file):
        self.bin_file = bin_file
        if not os.path.exists(bin_file):
            raise Exception("File not exit: " + bin_file)
        self.bin_data = None

    def load(self):
        def __parse_bin_name(bin_file):
            filename = os.path.basename(bin_file)
            filename = os.path.splitext(filename)[0]
            infos = filename.split('_')
            # ops_info = infos[0]  # Conv2d--gradConv2D--Conv2DBackpropFilter-op865
            # ops_direct = infos[1]  # output
            # ops_idx = infos[2]  # 0
            ops_shape = infos[4:-2]  # [32, 4, 16, 16]
            ops_dtype = infos[-2]  # Float32
            return tuple(int(i) for i in ops_shape), TYPE_MAPPER[ops_dtype]

        bin_shape, bin_dtype = __parse_bin_name(self.bin_file)
        print(">>> parse bin name ok ~")
        print(">>> -- bin_shape = {}".format(bin_shape))
        print(">>> -- bin_dtype = {}".format(bin_dtype))

        self.bin_data = np.fromfile(self.bin_file, bin_dtype).reshape(bin_shape)
        print(">>> load bin to numpy success~")
        return self

    def is_nan_exist(self):
        if self.bin_data is None:
            print("Please call load() first!!!")
            return

        def __is_nan_exist(bin_data):
            if len(bin_data.shape) > 1:
                for item in bin_data:
                    if __is_nan_exist(item):
                        return True
            if np.isnan(bin_data).any():
                print(">>>>>>>> is nan", bin_data)
                return True
            return False

        return __is_nan_exist(self.bin_data)

    def print_bin(self):
        def __print_numpy_array(numpy_array, width=16):
            if len(numpy_array.shape) > 1:
                for item in numpy_array:
                    __print_numpy_array(item)

            height = int(len(numpy_array) / width) + 1
            for i in range(height):
                star_idx = i * width
                end_idx = (star_idx + width) if (star_idx + width) < len(numpy_array) else len(numpy_array)
                tmp_data = numpy_array[star_idx: end_idx]
                print("[{}:{}] -> {}".format(star_idx, end_idx, tmp_data))

        __print_numpy_array(self.bin_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bin-file', type=str, default="./my.bin", help="bin file")

    args, _ = parser.parse_known_args()

    loader = BinLoader("")
    loader.load()
    loader.is_nan_exist()
    loader.print_bin()
    loader.print_bin()

