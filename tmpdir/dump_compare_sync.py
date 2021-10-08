npy = "/home/cj/dump_sync/rank_0/FCN8s/1/0/Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633702468784428.input.0.DefaultFormat.npy"
npy = "/home/cj/dump_sync/rank_0/FCN8s/1/0/Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633702468786013.input.1.DefaultFormat.npy"
npy = "/home/cj/dump_sync/rank_0/FCN8s/1/0/Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633703088299848.input.1.DefaultFormat.npy"

import numpy as np

np_data = np.load(npy)
print(np_data)
print(np_data.shape)