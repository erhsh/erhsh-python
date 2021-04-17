import os

import erhsh.ms as ems


def create_dataset(batch_size=32):
    # import
    import mindspore.dataset.engine as de
    import numpy as np
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
    class BaseDataset:
        def __init__(self):
            self.samples = []
            self._load_samples()

        def __getitem__(self, index):
            sample = self.samples[index]
            return sample[0], sample[1]

        def _load_samples(self):
            self.samples.append(
                [np.random.rand(3, 4, 5).astype(np.float32),
                 np.random.randint(10, size=()).astype(np.int32)])

        def __len__(self):
            return len(self.samples)

    # define dataset
    ds = de.GeneratorDataset(source=BaseDataset(), column_names=['image', 'label'],
                             num_shards=num_shards, shard_id=shard_id)

    # map ops
    ds = ds.map(input_columns=["image"], operations=lambda img: img, num_parallel_workers=8)

    # batch & repeat
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    ds = ds.repeat(count=1)

    return ds


if __name__ == '__main__':
    dataset = create_dataset()
    ems.print_ds_info(dataset)
    ems.print_ds_data(dataset)
