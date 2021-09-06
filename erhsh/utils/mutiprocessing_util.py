from multiprocessing import Process, Manager
import numpy as np
import os
import inspect


class MutiProcessor():
    def __init__(self, np_data, data_name="", h_num=1, w_num=1, cube_func=None):
        if len(np_data.shape) < 2:
            raise Exception("np_data must at least have dim 2.")
        self.np_data = np_data
        self.data_name = data_name

        self.h_size, self.w_size = np_data.shape[:2]
        print(">>>>>>>-- self.h_size", self.h_size, ", self.w_size=", self.w_size)

        if h_num > self.h_size or w_num > self.w_size:
            raise Exception("h_num or w_num must at less then np_data shape {}. but got {}".format(np_data.shape[:2], (h_num, w_num)))
        self.h_num, self.w_num = h_num, w_num
        print(">>>>>>>-- self.h_num", self.h_num, ", self.w_num=", self.w_num)

        self.h_step, self.w_step = self.h_size // self.h_num, self.w_size // self.w_num
        print(">>>>>>>-- self.h_step", self.h_step, ", self.w_step=", self.w_step)

        self.__cube_func = cube_func
        self.__ret_dict = Manager().dict()

    def __get_location(self, i, j):
        start_h, end_h = i * self.h_step, (i + 1) * self.h_step
        start_w, end_w = j * self.w_step, (j + 1) * self.w_step
        end_h = end_h if end_h < self.h_size else self.h_size
        end_w = end_w if end_w < self.w_size else self.w_size
        return (start_h, start_w, end_h, end_w)

    def __cube_process(self, i, j):
        start_h, start_w, end_h, end_w = self.__get_location(i, j)
        cube_input = self.np_data[start_h:end_h, start_w:end_w]
        print(">>>>>>>>>__cube_process: ({},{}) = {} start...".format(i, j, cube_input.shape))

        cube_output = None
        if self.__cube_func:
            func_args = inspect.getfullargspec(self.__cube_func).args
            if len(func_args) == 1:
                cube_output = self.__cube_func(cube_input)
            elif len(func_args) == 2:
                cube_output = self.__cube_func(cube_input, (start_h, start_w, end_h, end_w))
            else:
                raise Exception("args num invalid." + str(func_args))
        print(">>>>>>>>>__cube_process: ({},{}) = {} end.".format(i, j, cube_output.shape if cube_output is not None else None))

        key = "_".join([str(x) for x in (start_h, start_w, end_h, end_w)])
        self.__ret_dict[key] = cube_output

    def process(self):
        _process_list = []
        for i in range(self.h_num):
            for j in range(self.w_num):
                p = Process(target=self.__cube_process, args=(i, j))
                p.start()
                _process_list.append(p)

        for p in _process_list:
            p.join()

        return self.__ret_dict


if __name__ == 'main':
    img_np = np.arange(10000).reshape(100, 100)
    def cube_func(cube_input, locs):
        pass
    processor = MutiProcessor(img_np, h_num=5, w_num=5, cube_func=gray2RGB)
    ret = processor.process()