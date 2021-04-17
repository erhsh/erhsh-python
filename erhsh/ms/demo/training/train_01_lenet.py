import os

import erhsh.ms as ems

parser = ems.ArgumentParser()
args, _ = parser.parse_known_args()


# # train test config
# args.data_path = "./mnist"
# args.batch_size = 32
# args.device_target = "Ascend"
# args.graph_mode = True
# args.epoch_size = 10
#
# # eval test config
# args.ckpt_path = "./ckpt/lenet-10_1875.ckpt"
# args.do_eval = True

# print("parser:", parser)
# print("args:", args)


def main():
    from mindspore import nn

    # dataset
    if os.path.exists(args.data_path):
        dataset = ems.create_mnist_dataset(args.data_path, is_train=(not parser.do_eval), batch_size=args.batch_size)
    else:
        dataset = ems.create_mocker_dataset()
    ems.print_ds_info(dataset)

    # network
    network = ems.LeNet5()

    # loss & opt
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    # train
    ems.Trainer('lenet', parser) \
        .set_dataset(dataset) \
        .set_network(network) \
        .set_loss_fn(loss_fn) \
        .set_optimizer(optimizer) \
        .set_callbacks([ems.TimeMonitor()]) \
        .run()


if __name__ == '__main__':
    main()
