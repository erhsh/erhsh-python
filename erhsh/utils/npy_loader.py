import numpy as np

def load_npy(npy_path):
    print(">>> npy path:", npy_path)
    np_data = np.load(npy_path)
    print(">>> npy shape:", np_data.shape)
    print(">>> npy data: \n", np_data)



if __name__ == "__main__":
    npy_path="/home/cj/dump_sync/rank_0/FCN8s/1/0/Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633703088299848.input.1.DefaultFormat.npy"
    load_npy(npy_path=npy_path)
    pass