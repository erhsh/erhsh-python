import inspect

import numpy as np


class DatasetMocker(object):
    def __init__(self, data_len=256, batch_size=32, repeat_count=1, features=None, feature_names=None, npw=8):
        self.__data_len = data_len
        self.__batch_size = batch_size
        self.__repeat_count = repeat_count
        self.__features = features
        self.__feature_names = feature_names
        self.__npw = npw

    def set_data_len(self, data_len):
        self.__data_len = data_len
        return self

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size
        return self

    def set_repeat_count(self, repeat_count):
        self.__repeat_count = repeat_count
        return self

    def set_features(self, features):
        self.__features = features
        return self

    def set_feature_names(self, feature_names):
        self.__feature_names = feature_names
        return self

    def set_npw(self, npw):
        self.__npw = npw
        return self

    def mock(self):
        if self.__features is None:
            self.__features = {
                'image': lambda row_idx: np.ones((3, 100, 100), dtype=np.int32) * row_idx % 256,
                'label': lambda row_idx: np.array(np.ones((), dtype=np.int32) * row_idx % 256)
            }

        if self.__feature_names is None:
            if isinstance(self.__features, dict):
                self.__feature_names = list(self.__features.keys())
            elif isinstance(self.__features, (list, tuple)):
                self.__feature_names = ['feature_' + str(i) for i, _ in enumerate(self.__features)]
            else:
                self.__feature_names = ['undefined']

        def ds_gen_func():
            for row_idx in range(self.__data_len):
                if isinstance(self.__features, (dict, list, tuple)):
                    feature_values = self.__features
                    if isinstance(self.__features, dict):
                        feature_values = self.__features.values()
                    yield tuple(
                        v if not callable(v) else v(row_idx) if len(inspect.getfullargspec(v).args) == 1 else v()
                        for v in feature_values)
                else:
                    yield tuple([self.__features])

        import mindspore.dataset as de
        dataset = de.GeneratorDataset(ds_gen_func, column_names=self.__feature_names, num_parallel_workers=self.__npw)
        dataset = dataset.batch(batch_size=self.__batch_size, drop_remainder=True)
        dataset = dataset.repeat(self.__repeat_count)

        return dataset
