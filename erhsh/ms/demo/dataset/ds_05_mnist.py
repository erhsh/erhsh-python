import os

import erhsh.ms as ems


def create_dataset(data_path, is_train=True, batch_size=32):
    # import
    import mindspore.common.dtype as mstype
    import mindspore.dataset.engine as de
    import mindspore.dataset.transforms.c_transforms as C2
    import mindspore.dataset.vision.c_transforms as C
    from mindspore.common import set_seed
    from mindspore.dataset.vision import Inter

    set_seed(1)

    # shard
    num_shards = shard_id = None
    rand_size = os.getenv("RANK_SIZE")
    rand_id = os.getenv("RANK_ID")
    if rand_size is not None and rand_id is not None:
        num_shards = int(rand_size)
        shard_id = int(rand_id)

    # define dataset
    data_path = os.path.join(data_path, "train" if is_train else "test")
    ds = de.MnistDataset(data_path, num_shards=num_shards, shard_id=shard_id)

    # define ops
    comps_ops = list()
    comps_ops.append(C.Resize((32, 32), interpolation=Inter.LINEAR))
    comps_ops.append(C.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081))
    comps_ops.append(C.Rescale(1 / 255., 0.))
    comps_ops.append(C.HWC2CHW())

    # map ops
    ds = ds.map(input_columns=["image"], operations=comps_ops, num_parallel_workers=8)
    ds = ds.map(input_columns=["label"], operations=C2.TypeCast(mstype.int32), num_parallel_workers=8)

    # batch & repeat
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=batch_size, drop_remainder=is_train)
    ds = ds.repeat(count=1)

    return ds


if __name__ == '__main__':
    dataset = create_dataset("D:/Data/dataset/mnist", is_train=True)
    ems.print_ds_info(dataset)
    ems.print_ds_data(dataset)
