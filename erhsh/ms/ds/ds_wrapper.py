from collections import OrderedDict

import numpy as np

from erhsh.utils import create_dir


class DatasetWrapper(object):
    def __init__(self, ds):
        from mindspore.dataset.engine import BatchDataset, RepeatDataset
        assert isinstance(ds, (BatchDataset, RepeatDataset))
        self.ds = ds
        self.odict = OrderedDict()
        self.__parse_ds()

    def __parse_ds(self):
        for x in self.ds.create_dict_iterator():
            for k in x.keys():
                v = np.array([x.get(k).asnumpy()])
                if k not in self.odict:
                    self.odict[k] = v
                else:
                    exist_v = self.odict[k]
                    self.odict[k] = np.vstack((exist_v, v))

    def __getitem__(self, item):
        ret = self.odict.get(item)
        if ret is not None:
            return ret

        if isinstance(item, int):
            return self.odict.get(list(self.odict.keys())[item])

        return None

    def ds_info(self, prefix="|-"):
        print("{}get_dataset_size={}".format(prefix, self.ds.get_dataset_size()), flush=True)
        print("{}get_repeate_count={}".format(prefix, self.ds.get_repeat_count()), flush=True)
        print("{}output_types={}".format(prefix, self.ds.output_types()), flush=True)
        print("{}output_shapes={}".format(prefix, self.ds.output_shapes()), flush=True)

        for x in self.ds.create_dict_iterator():
            keys = x.keys()
            for k in keys:
                v = x.get(k)
                print("{} {} -> {} {}".format(prefix, k, v.shape, v.dtype), flush=True)
            break


# store images
CACHE_DIR = '.cache_ds'


class ImageDatasetWrapper(DatasetWrapper):
    def __init__(self, ds, img_type="CHW"):
        super().__init__(ds)
        self.img_store_dir = CACHE_DIR
        self.img_type = img_type

    def save_as_img(self, img_key=None, img_store_dir=None):
        if img_store_dir is not None:
            self.img_store_dir = img_store_dir

        xnchw = self.__getitem__(img_key)
        if xnchw is None:
            raise RuntimeError("Key not found : " + img_key)

        self.__conv2pic(xnchw)

    def __getitem__(self, item):
        return super(ImageDatasetWrapper, self).__getitem__(item)

    def __conv2pic(self, xnchw):
        assert xnchw.ndim == 5
        img_store_dir = create_dir(self.img_store_dir)
        for i, nchw in enumerate(xnchw):
            batch_store_dir = create_dir('{}/batch-{}'.format(img_store_dir, i))
            for j, chw in enumerate(nchw):
                image_name = '{}/{}_{}.jpg'.format(batch_store_dir, j, '-'.join((str(x) for x in chw.shape)))
                self.__conv2pic0(chw, image_name)

    def __conv2pic0(self, chw, image_name):
        assert chw.ndim == 3
        if self.img_type == "CHW":
            hwc = np.transpose(chw, (2, 1, 0))
            if hwc.dtype == np.int32:
                hwc = hwc.astype(np.uint8)
        else:
            hwc = chw

        import matplotlib.pyplot as plt
        plt.imsave(image_name, hwc)
