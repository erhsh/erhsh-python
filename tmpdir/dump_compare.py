
import numpy as np
import os

from numpy.lib.utils import info

base_dir = "/home/cj/dump/"

def get_dump_infos(op_prefix, ops, op_sufix):
    ops = ops.strip().split()
    op_sufix = op_sufix.strip().split()
    infos = []
    for i, sufix in enumerate(op_sufix):
        info = []
        for j in range(len(ops)):
            tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix))
            if j == 0:
                tmp = tmp # left
            elif j == len(ops) - 1:
                tmp = tmp # right
            info.append(tmp)
        infos.append(info)
    return infos


if __name__ == "__main__":
    '''
    step1: export MINDSPORE_DUMP_CONFIG=/home/cj/data_dump.json
    step2: grep "Conv2DTranspose" rank_*/execution_order/ms_execution_order_graph_1.csv | grep -v "Gradients"
    step3: find . -name "*Conv2DTranspose-op301*"
    step4: python /usr/local/Ascend/ascend-toolkit/5.0.T302/arm64-linux/tools/operator_cmp/compare/msaccucmp.py convert -t npy -f NCHW -d 
    '''

    op_prefix = "ApplyMomentum.Default_network-TrainOneStepWithLossScaleCell_optimizer-Momentum_ApplyMomentum-"

    ops = """
    op402.240.9.1632987180762438
    op458.265.38.1632987180764818
    op458.265.87.1632987180768536
    op458.265.87.1632987180765500
    op458.265.87.1632987180910887
    op458.265.87.1632987180899365
    op458.265.87.1632987180911145
    op458.265.30.1632987180907789

    op458.265.5.1632987180888982
    op458.265.34.1632987180877431
    op458.265.34.1632987180882284
    op458.265.34.1632987180878487
    op458.265.34.1632987180861535
    op458.265.34.1632987180858725
    op458.265.34.1632987180864142
    op458.265.34.1632987180861231

    op458.265.4.1632986291560048
    op458.265.30.1632986291556854
    op458.265.30.1632986291562342
    op458.265.30.1632986291559862
    op458.265.30.1632986291557703
    op458.265.30.1632986291552056
    op458.265.30.1632986291557328
    op458.265.30.1632986291553589

    op458.265.4.1632986929231316
    op458.265.30.1632986929228161
    op458.265.30.1632986929228909
    op458.265.30.1632986929226465
    op458.265.30.1632986929235089
    op458.265.30.1632986929227748
    op458.265.30.1632986929231300
    op402.240.6.1632986929233234

    """

    op_sufix = """
    .input.0.npy
    .input.1.npy
    .input.2.npy
    .input.3.npy
    .input.4.npy
    .output.0.npy
    .output.1.npy
    """

    infos = get_dump_infos(op_prefix=op_prefix, ops=ops, op_sufix=op_sufix)
    info_a = infos[0][0]
    print("info_a: ", info_a.shape)
    print(info_a)
    
    ops = """
    op248.125.2.1632988441630167
    """

    infos = get_dump_infos(op_prefix=op_prefix, ops=ops, op_sufix=op_sufix)
    info_b = infos[0][0]
    print("info_b: ", info_b.shape)
    print(info_b)

    print(np.allclose(info_a, info_b, rtol=1e-2, atol=1e-2))

    # print(np.load(os.path.join(base_dir, op_prefix + "op248.125.2.1632988441630167.input.0.npy")))