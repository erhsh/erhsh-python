
import numpy as np
import os


np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=1*6*512*16)
# np.set_printoptions(edgeitems=32)
base_dir = "/home/cj/dump_sync/"

#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703396227322.input.0.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703396227552.input.1.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703396227855.input.2.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703396228013.input.3.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703396228178.input.4.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703399391562.output.0.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op248.0.0.1633703399391750.output.1.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618129.input.0.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618345.input.1.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618542.input.2.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618647.input.3.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618778.input.4.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088924086.output.0.DefaultFormat.npy
#  ApplyMomentum.ApplyMomentum-op402.0.0.1633703088924269.output.1.DefaultFormat.npy

path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618129.input.0.DefaultFormat.npy"
# path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618345.input.1.DefaultFormat.npy"
# path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618542.input.2.DefaultFormat.npy"
# path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618647.input.3.DefaultFormat.npy"
# path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088618778.input.4.DefaultFormat.npy"
# path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088924086.output.0.DefaultFormat.npy"
# path_a = "ApplyMomentum.ApplyMomentum-op402.0.0.1633703088924269.output.1.DefaultFormat.npy"

path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703396227322.input.0.DefaultFormat.npy"
# path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703396227552.input.1.DefaultFormat.npy"
# path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703396227855.input.2.DefaultFormat.npy"
# path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703396228013.input.3.DefaultFormat.npy"
# path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703396228178.input.4.DefaultFormat.npy"
# path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703399391562.output.0.DefaultFormat.npy"
# path_b = "ApplyMomentum.ApplyMomentum-op248.0.0.1633703399391750.output.1.DefaultFormat.npy"



path_a = "Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633703088297579.input.0.DefaultFormat.npy"
path_a = "Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633703088299848.input.1.DefaultFormat.npy"
# path_a = "Conv2DBackpropInput.Conv2DTranspose-op251.0.0.1633703088752666.output.0.DefaultFormat.npy"

path_b = "Conv2DBackpropInput.Conv2DTranspose-op132.0.0.1633703392872927.input.0.DefaultFormat.npy"
path_b = "Conv2DBackpropInput.Conv2DTranspose-op132.0.0.1633703392901732.input.1.DefaultFormat.npy"
# path_b = "Conv2DBackpropInput.Conv2DTranspose-op132.0.0.1633703397858813.output.0.DefaultFormat.npy"



info_a = np.load(os.path.join(base_dir, path_a))[:,:,:,:8]
info_b = np.load(os.path.join(base_dir, path_b))[:,:,:,:8]

print("info_a: ", info_a.shape, info_a.dtype)
print(info_a)
print("info_b: ", info_b.shape, info_b.dtype)
print(info_b)
print(np.allclose(info_a, info_b, rtol=1e-5, atol=1e-5))

