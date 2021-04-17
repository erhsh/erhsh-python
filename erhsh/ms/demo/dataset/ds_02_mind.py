import os

import erhsh.ms as ems


def create_dataset(data_path, is_train=True, batch_size=32):
    # import
    import mindspore.dataset.engine as de
    import mindspore.dataset.vision.c_transforms as C
    from mindspore.common import set_seed

    set_seed(1)

    # shard
    num_shards = shard_id = None
    rand_size = os.getenv("RANK_SIZE")
    rand_id = os.getenv("RANK_ID")
    if rand_size is not None and rand_id is not None:
        num_shards = int(rand_size)
        shard_id = int(rand_id)

    # define dataset
    ds = de.MindDataset(data_path, columns_list=['data'], shuffle=True,
                        num_shards=num_shards, shard_id=shard_id,
                        num_parallel_workers=8, num_samples=None)

    # map ops
    ds = ds.map(input_columns=["data"], operations=C.Decode())
    ds = ds.map(input_columns=["data"], operations=C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255]))
    ds = ds.map(input_columns=["data"], operations=C.Resize((224, 224)))
    ds = ds.map(input_columns=["data"], operations=C.HWC2CHW())

    # batch & repeat
    ds = ds.batch(batch_size=batch_size, drop_remainder=is_train)
    ds = ds.repeat(count=1)

    return ds


if __name__ == '__main__':
    dataset = create_dataset("D:/Data/dataset/voc2012/t1/t1.m0", is_train=True)
    ems.print_ds_info(dataset)
    ems.print_ds_data(dataset)
