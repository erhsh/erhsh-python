import numpy as np

import erhsh.ms as ems

np.random.seed(1)


def create_dataset():
    return ems.DatasetMocker().set_features({
        "data": lambda: np.random.rand(1, 32, 32).astype(np.float32),
        "label": lambda: np.random.randint(10, size=()).astype(np.int32)
    }).set_data_len(3 * 32).set_batch_size(32).mock()


if __name__ == '__main__':
    dataset = create_dataset()
    ems.print_ds_info(dataset)
    ems.print_ds_data(dataset)
