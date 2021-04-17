import erhsh
from erhsh.ms import create_dataset, print_ds_info, print_ds_performance, print_ds_data
from erhsh.ms.tools.hccl_tool import gen_rank_table_file
from erhsh.utils import file_util


def test_01_file_util():
    print("sub dir is:", file_util.list_sub_dirs("../"))


def test_02_create_dataset():
    ds = create_dataset()
    print(">> create ds success:", ds)


def test_03_ds_print_func():
    ds = create_dataset()

    print(">>> print_ds_info:")
    print_ds_info(ds)

    print(">>> print_ds_performance:")
    print_ds_performance(ds)

    print(">>> print_ds_data:")
    print_ds_data(ds)


def test_04_gen_rank_table_file():
    server_id = "123.123.123.123"
    device_type = "A+K"
    rank_table = gen_rank_table_file(server_id, device_type)
    print("generate rank table:", rank_table)


if __name__ == '__main__':
    print("version is:", erhsh.__version__)
    test_01_file_util()
    test_02_create_dataset()
    test_03_ds_print_func()
    test_04_gen_rank_table_file()
