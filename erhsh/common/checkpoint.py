import abc
import os
import re

import numpy as np

from erhsh.utils.print_util import TblPrinter


class CheckpointLoader(object):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    @abc.abstractmethod
    def _load_checkpoint(self):
        pass

    def __list(self, filter_key=None):
        param_dict = self._load_checkpoint()

        filter_dict = {}
        for k, v in param_dict.items():
            if filter_key \
                    and (filter_key not in k) \
                    and (not re.search(filter_key, k, re.M | re.I)):
                continue
            filter_dict[k] = v

        return param_dict, filter_dict

    def __get(self, key):
        param_dict = self._load_checkpoint()
        return param_dict.get(key)

    def list(self, filter_key=None):
        param_dict, filter_dict = self.__list(filter_key=filter_key)

        ret = {}
        tp = TblPrinter("Param Keys", "Value Shape")
        for k, v in filter_dict.items():
            v = str(v.shape)
            ret[k] = v
            tp.add_row(k, v)
        tp.print()

        print("Filter/Total: {}/{}".format(len(ret), len(param_dict)))
        return ret

    def get(self, key):
        v = self.__get(key)
        if v is None:
            print("param key not found! key={}".format(key))
            return

        tp = TblPrinter("Param Keys", "Value Shape", "Value Type")
        tp.add_row(key, str(v.shape), str(v.dtype))
        vf = v.flatten()
        length = len(vf)
        if length <= 100:
            tp.add_row(vf)
        elif length <= 200:
            tp.add_row(vf[:100])
            tp.add_row(vf[100:])
        else:
            tp.add_row(vf[:100])
            tp.add_row("...")
            tp.add_row(vf[-100:])
        tp.print()

        print("Max:{:.7f}, Min:{:.7f}, Mean:{:.7f}".format(v.max(), v.min(), v.mean()))

    def list_dump(self, filter_key=None, dump_to=None):
        print("Begin dump {0} to {1}".format(self.checkpoint_path, dump_to))
        print("Filter is: {}".format(filter_key))

        _, filter_dict = self.__list(filter_key=filter_key)

        for k, v in filter_dict.items():
            dump_file = os.path.join(dump_to, k + '.npy')
            np.save(dump_file, v)
            print("Dump to: {}".format(dump_file))

    def get_dump(self, key, dump_to=None):
        print("Begin dump {0} to {1}".format(self.checkpoint_path, dump_to))
        print("Key is: {}".format(key))

        v = self.__get(key)
        dump_file = os.path.join(dump_to, key + '.npy')
        np.save(dump_file, v)
        print("Dump to: {}".format(dump_file))
