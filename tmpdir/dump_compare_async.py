
import numpy as np
import os


np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=64*256*256)
# np.set_printoptions(edgeitems=256)
base_dir = "/home/cj/dump/"
# base_dir = "/home/cj/dump.bak/"

def get_dump_infos(op_prefix, ops, op_sufix):
    ops = [x.strip() for x in ops.strip().split("\n")]
    op_sufix = [x.strip() for x in op_sufix.strip().split("\n")]
    infos = []
    for i, sufix in enumerate(op_sufix):
        info = []
        sufix = [x.strip() for x in sufix.split(",")]
        for j in range(len(ops)):
            if len(sufix) == 3:
                if j == 0: # left
                    tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[0]))[:,:,:,:-1]
                elif j == len(ops) - 1: # right
                    tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[2]))[:,:,:,1:]
                else: # middle
                    tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[1]))[:,:,:,1:-1]
            elif len(sufix) == 2:
                if j == 0: # left
                    tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[0]))[:,:,:,:-1]
                elif j == len(ops) - 1: # right
                    tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[0]))[:,:,:,1:]
                else: # middle
                    tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[1]))[:,:,:,1:-1]
            else:
                tmp = np.load(os.path.join(base_dir, op_prefix + ops[j] + sufix[0]))
            info.append(tmp)
        infos.append(info)
    return infos


def test_01_compare_ApplyMomentum():
    op_prefix_a = "ApplyMomentum.Default_network-TrainOneStepWithLossScaleCell_optimizer-Momentum_ApplyMomentum-"
    ops_a = "op402.240.15.1633694047077397" # weight is 0.01
    # ops_a = "op402.240.16.1633696994623399" # weight not 0.01
    op_sufix_a = """
    .input.0.npy
    .input.1.npy
    .input.2.npy
    .input.3.npy
    .input.4.npy
    .output.0.npy
    .output.1.npy
    """
    infos_a = get_dump_infos(op_prefix=op_prefix_a, ops=ops_a, op_sufix=op_sufix_a)
    info_a = infos_a[5][0]
    print(">>> info a: ", info_a.shape, info_a.dtype)
    print(info_a)


    op_prefix_b = "ApplyMomentum.Default_network-TrainOneStepWithLossScaleCell_optimizer-Momentum_ApplyMomentum-"
    ops_b = "op248.125.3.1633694074208709" # weight is 0.01
    # ops_b = "op248.125.3.1633697035510977" # weight not 0.01
    op_sufix_b = """
    .input.0.npy
    .input.1.npy
    .input.2.npy
    .input.3.npy
    .input.4.npy
    .output.0.npy
    .output.1.npy
    """
    infos_b = get_dump_infos(op_prefix=op_prefix_b, ops=ops_b, op_sufix=op_sufix_b)
    info_b = infos_b[5][0]
    print(">>> info b: ", info_b.shape, info_b.dtype)
    print(info_b)

    # compare
    print(np.allclose(info_a, info_b, rtol=1e-3, atol=1e-3))


def test_02_compare_Conv2DTranspose():
    op_prefix_a = "Conv2DBackpropInput.Default_network-TrainOneStepWithLossScaleCell_network-_VirtualDatasetCell__backbone-WithLossCell__backbone-FCN8s_upscore_pool3-Conv2dTranspose_Conv2DTranspose-"
    ops_a = """
    op251.87.15.1633694046582321
    op301.98.15.1633694046587832
    op301.98.15.1633694046587804
    op301.98.15.1633694046589671
    op301.98.15.1633694046577607
    op301.98.15.1633694046576869
    op301.98.15.1633694046576044
    op301.98.15.1633694046576999
    op301.98.15.1633694046580526
    op301.98.15.1633694046578684
    op301.98.15.1633694046578767
    op301.98.15.1633694046579180
    op301.98.15.1633694046527597
    op301.98.15.1633694046528995
    op301.98.15.1633694046527292
    op301.98.15.1633694046528031
    op301.98.15.1633693172963993
    op301.98.15.1633693172964152
    op301.98.15.1633693172964186
    op301.98.15.1633693172962193
    op301.98.15.1633693172982863
    op301.98.15.1633693172982743
    op301.98.15.1633693172982542
    op301.98.15.1633693172982352
    op301.98.15.1633693821836712
    op301.98.15.1633693821836106
    op301.98.15.1633693821836817
    op301.98.15.1633693821836726
    op301.98.15.1633693821832175
    op301.98.15.1633693821832468
    op301.98.15.1633693821830917
    op251.87.15.1633693821826492
    """
    ops_a = """
    op251.87.16.1633696994105938
    op301.98.16.1633696994111008
    op301.98.16.1633696994112047
    op301.98.16.1633696994110107
    op301.98.16.1633696994095710
    op301.98.16.1633696994093776
    op301.98.16.1633696994094567
    op301.98.16.1633696994094508
    op301.98.16.1633696994101890
    op301.98.16.1633696994101159
    op301.98.16.1633696994100792
    op301.98.16.1633696994101547
    op301.98.16.1633696994035319
    op301.98.16.1633696994035346
    op301.98.16.1633696994035787
    op301.98.16.1633696994035658
    op301.98.16.1633696120458165
    op301.98.16.1633696120459295
    op301.98.16.1633696120458600
    op301.98.16.1633696120457637
    op301.98.16.1633696120476538
    op301.98.16.1633696120476691
    op301.98.16.1633696120476240
    op301.98.16.1633696120478317
    op301.98.16.1633696769282286
    op301.98.16.1633696769282687
    op301.98.16.1633696769282450
    op301.98.16.1633696769281340
    op301.98.16.1633696769338017
    op301.98.16.1633696769338518
    op301.98.16.1633696769336774
    op251.87.16.1633696769331244
    """
    op_sufix_a = """
    .input.0.1x64x256x9.npy, .input.0.1x64x256x10.npy, .input.0.1x64x256x9.npy
    .input.1.64x6x4x4.npy
    .output.0.1x6x512x16.npy
    """
    infos_a = get_dump_infos(op_prefix=op_prefix_a, ops=ops_a, op_sufix=op_sufix_a)
    info_a = np.concatenate(infos_a[0], axis=3)
    info_a = infos_a[1][0]
    info_a = np.concatenate(infos_a[2], axis=3)
    print(">>> info a: ", info_a.shape, info_a.dtype)
    print(info_a)


    op_prefix_b = "Conv2DBackpropInput.Default_network-TrainOneStepWithLossScaleCell_network-WithLossCell__backbone-FCN8s_upscore_pool3-Conv2dTranspose_Conv2DTranspose-"
    ops_b = "op132.48.3.1633694070059526" # weight is 0.01
    ops_b = "op132.48.3.1633697031309592" # weight not 0.01
    op_sufix_b = """
    .input.0.1x64x256x256.npy
    .input.1.64x6x4x4.npy
    .output.0.1x6x512x512.npy
    """

    infos_b = get_dump_infos(op_prefix=op_prefix_b, ops=ops_b, op_sufix=op_sufix_b)
    info_b = infos_b[0][0]
    info_b = infos_b[1][0]
    info_b = infos_b[2][0]
    print(">>> info b: ", info_b.shape, info_b.dtype)
    print(info_b)

    # compare
    print(np.allclose(info_a, info_b, rtol=1e-3, atol=1e-3))


def test_03_compare_SoftmaxCrossEntropyWithLogits():
    op_prefix_a = "SoftmaxCrossEntropyWithLogits_SoftmaxCrossEntropyWithLogits-"
    ops_a = """
    op259.97.15.1633694046615755
    op309.108.15.1633694046615957
    op309.108.15.1633694046615704
    op309.108.15.1633694046615887
    op309.108.15.1633694046604135
    op309.108.15.1633694046603795
    op309.108.15.1633694046604104
    op309.108.15.1633694046603779
    op309.108.15.1633694046607140
    op309.108.15.1633694046606570
    op309.108.15.1633694046606664
    op309.108.15.1633694046606374
    op309.108.15.1633694046555684
    op309.108.15.1633694046555645
    op309.108.15.1633694046555481
    op309.108.15.1633694046555454
    op309.108.15.1633693172991682
    op309.108.15.1633693172991779
    op309.108.15.1633693172991534
    op309.108.15.1633693172991324
    op309.108.15.1633693173010727
    op309.108.15.1633693173010747
    op309.108.15.1633693173010689
    op309.108.15.1633693173010371
    op309.108.15.1633693821864683
    op309.108.15.1633693821865004
    op309.108.15.1633693821864889
    op309.108.15.1633693821864707
    op309.108.15.1633693821859679
    op309.108.15.1633693821859781
    op309.108.15.1633693821859389
    op259.97.15.1633693821859562
    """
    # ops_a = """
    # op259.97.16.1633696994139640
    # op309.108.16.1633696994139856
    # op309.108.16.1633696994139731
    # op309.108.16.1633696994139118
    # op309.108.16.1633696994123754
    # op309.108.16.1633696994123092
    # op309.108.16.1633696994123486
    # op309.108.16.1633696994123061
    # op309.108.16.1633696994130957
    # op309.108.16.1633696994130368
    # op309.108.16.1633696994130764
    # op309.108.16.1633696994130288
    # op309.108.16.1633696994064462
    # op309.108.16.1633696994064405
    # op309.108.16.1633696994064424
    # op309.108.16.1633696994064843
    # op309.108.16.1633696120486789
    # op309.108.16.1633696120486999
    # op309.108.16.1633696120485778
    # op309.108.16.1633696120485713
    # op309.108.16.1633696120505903
    # op309.108.16.1633696120505914
    # op309.108.16.1633696120505767
    # op309.108.16.1633696120505455
    # op309.108.16.1633696769310443
    # op309.108.16.1633696769310754
    # op309.108.16.1633696769310801
    # op309.108.16.1633696769310669
    # op309.108.16.1633696769366709
    # op309.108.16.1633696769366800
    # op309.108.16.1633696769366483
    # op259.97.16.1633696769366472
    # """
    op_sufix_a = """
    .input.0.8192x6.npy  
    .input.1.8192x6.npy  
    .output.0.8192.npy   
    .output.1.8192x6.npy 
    """
    infos_a = get_dump_infos(op_prefix=op_prefix_a, ops=ops_a, op_sufix=op_sufix_a)
    info_a = np.concatenate(infos_a[0], axis=0)
    info_a = np.concatenate(infos_a[1], axis=0)
    info_a = np.concatenate(infos_a[2], axis=0)
    info_a = np.concatenate(infos_a[3], axis=0)
    print(">>> info a: ", info_a.shape, info_a.dtype)
    print(info_a)


    op_prefix_b = "SoftmaxCrossEntropyWithLogits_SoftmaxCrossEntropyWithLogits-"
    ops_b = "op138.52.3.1633694070183490" # weight is 0.01
    # ops_b = "op138.52.3.1633697031432840" # weight not 0.01
    op_sufix_b = """
    .input.0.262144x6.npy
    .input.1.262144x6.npy
    .output.0.262144.npy
    .output.1.262144x6.npy
    """

    infos_b = get_dump_infos(op_prefix=op_prefix_b, ops=ops_b, op_sufix=op_sufix_b)
    info_b = infos_b[0][0]
    info_b = infos_b[1][0]
    info_b = infos_b[2][0]
    info_b = infos_b[3][0]
    print(">>> info b: ", info_b.shape, info_b.dtype)
    print(info_b)

    # compare
    print(np.allclose(info_a, info_b, rtol=1e-5, atol=1e-5))



if __name__ == "__main__":
    # test_01_compare_ApplyMomentum()
    # test_02_compare_Conv2DTranspose()
    test_03_compare_SoftmaxCrossEntropyWithLogits()