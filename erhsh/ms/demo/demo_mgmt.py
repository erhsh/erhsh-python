import json
import os
import shutil
import sys

import erhsh.utils as eut

DEFAULT_FETCH_TO = "emsd"


class DemoMgmt:
    def __init__(self, name=None, dir_name="demo"):
        self.__name = name
        self.__mgmt_path = self.__get_mgmt_path()
        self.__demo_dir_path = os.path.join(self.__mgmt_path, dir_name)

    def __get_mgmt_path(self):
        return os.path.dirname(os.path.realpath(sys.modules[self.__module__].__file__))

    def __demo_list(self):
        rst = [f for f in os.listdir(self.__demo_dir_path) if
               os.path.isfile(os.path.join(self.__demo_dir_path, f)) and '__init__.py' != f]
        rst.sort()
        return rst

    def print_demo_list(self):
        tp = eut.TblPrinter("ID", "File Name")
        for i, fn in enumerate(self.__demo_list()):
            tp.add_row(str(i), fn)
        tp.print()

    def __find_demo(self, key):
        fs = self.__demo_list()

        fid = eut.safe2int(key)
        if isinstance(fid, int) and fid < len(fs):
            return fs[fid]

        if key in fs:
            return key

        return None

    def __gen_jupyter_file(self, dest_file):
        with open(os.path.join(self.__mgmt_path, 'jupyter_template.ipynb'), mode='r') as f:
            data = json.load(f)
        data['cells'][0]['source'].append('%load ' + os.path.basename(dest_file))

        jupyter_dest_file = dest_file.replace('.py', '.ipynb')
        with open(jupyter_dest_file, mode='w') as f:
            json.dump(data, f, indent=4)

        return jupyter_dest_file

    def fetch_demo(self, key, fetch_to=None, view=False):
        os.system("")
        file_name = self.__find_demo(key)
        if file_name is None:
            print(">>> Please use The `ID' or 'File Name' from table blow!")
            self.print_demo_list()
            return
        src_file = os.path.join(self.__demo_dir_path, file_name)

        if view:
            file_gen = (row for row in open(src_file))
            for line in file_gen:
                print("\033[1;92m" + line + "\033[0m", end="")
            return

        if fetch_to is None:
            fetch_to = '_'.join([DEFAULT_FETCH_TO, self.__name]) if DEFAULT_FETCH_TO else DEFAULT_FETCH_TO
        fetch_to = fetch_to if fetch_to.startswith("/") or ":" in fetch_to else os.path.join(os.getcwd(), fetch_to)
        os.makedirs(fetch_to, exist_ok=True)

        dest_file = os.path.join(fetch_to, file_name)
        if os.path.exists(dest_file):
            print("\033[33m************* Warning: File Exist. Override It! ************\033[0m")
        shutil.copy(src_file, dest_file)
        print("\033[1;37;104mCopy To:\033[0m \033[1;92m{}\033[0m".format(dest_file))

        if dest_file.endswith('.py'):
            jupyter_dest_file = self.__gen_jupyter_file(dest_file)
            print("\033[1;37;104mGenerate .ipynb:\033[0m \033[1;92m{}\033[0m".format(jupyter_dest_file))


ds_mgmt = DemoMgmt(name="ds", dir_name="dataset")
net_mgmt = DemoMgmt(name="net", dir_name="network")
ops_mgmt = DemoMgmt(name="ops", dir_name="operator")
res_mgmt = DemoMgmt(name="res", dir_name="resource")
train_mgmt = DemoMgmt(name="train", dir_name="training")

if __name__ == '__main__':
    ds_mgmt.print_demo_list()
    net_mgmt.print_demo_list()
    ops_mgmt.print_demo_list()
    res_mgmt.print_demo_list()
    train_mgmt.print_demo_list()
