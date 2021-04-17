import os
import sys

import erhsh.utils as eut
from erhsh.ms.tools.hccl_tool_v1 import gen_rank_table_file
from erhsh.ms.train._argparse import ArgumentParser


class Trainer(object):
    def __init__(self, name=None, parser: ArgumentParser = None):
        self.__name = 'net' if name is None else name
        self.__parser = ArgumentParser() if parser is None else parser
        self.__parser.parse_known_args()

        # train needed
        self.__dataset = None
        self.__network = None
        self.__loss_fn = None
        self.__optimizer = None
        self.__metrics = None
        self.__callbacks = None

        # train params
        self.__do_eval = self.__parser.do_eval
        self.__epoch_size = self.__parser.epoch_size
        self.__do_sink = self.__parser.do_sink
        self.__sink_size = self.__parser.sink_size
        self.__ckpt_path = self.__parser.ckpt_path
        self.__device_target = self.__parser.device_target
        self.__graph_mode = self.__parser.graph_mode
        self.__save_graphs = self.__parser.save_graphs
        self.__visible_devices = self.__parser.visible_devices

    # train needed setters
    def set_dataset(self, val):
        self.__dataset = val
        return self

    def set_network(self, val):
        self.__network = val
        return self

    def set_loss_fn(self, val):
        self.__loss_fn = val
        return self

    def set_optimizer(self, val):
        self.__optimizer = val
        return self

    def set_metrics(self, val):
        self.__metrics = val
        return self

    def set_callbacks(self, val):
        self.__callbacks = val
        return self

    # train params setters
    def set_do_eval(self, val):
        self.__do_eval = val
        return self

    def set_epoch_size(self, val):
        self.__epoch_size = val
        return self

    def set_do_sink(self, val):
        self.__do_sink = val
        return self

    def set_sink_size(self, val):
        self.__sink_size = val
        return self

    def set_ckpt_path(self, val):
        self.__ckpt_path = val
        return self

    def set_device_target(self, val):
        self.__device_target = val
        return self

    def set_graph_mode(self, val):
        self.__graph_mode = val
        return self

    def set_save_graphs(self, val):
        self.__save_graphs = val
        return self

    def set_visible_devices(self, val):
        self.__visible_devices = val
        return self

    def __update_cbs(self):
        if not isinstance(self.__callbacks, list):
            return

        from mindspore.train.callback import LossMonitor, ModelCheckpoint

        loss_cb_exist, ckpt_cb_exist = False, False
        for cb in self.__callbacks:
            loss_cb_exist = isinstance(cb, LossMonitor)
            ckpt_cb_exist = isinstance(cb, ModelCheckpoint)

        if not loss_cb_exist:
            self.__callbacks.append(LossMonitor())

        if not ckpt_cb_exist:
            self.__callbacks.append(ModelCheckpoint(prefix=self.__name, directory=self.__ckpt_path))

    def __train(self):
        # import
        from mindspore import Model, load_param_into_net, load_checkpoint

        # load params
        if self.__ckpt_path and os.path.isfile(self.__ckpt_path) and os.path.exists(self.__ckpt_path):
            load_param_into_net(self.__network, load_checkpoint(self.__ckpt_path))

        # loss_fn & optimizer & metrics
        model = Model(self.__network, loss_fn=self.__loss_fn, optimizer=self.__optimizer, metrics=self.__metrics)

        # train
        print(">>>>>>>>>>>>>>>>>>>>> train start ... <<<<<<<<<<<<<<<<<<<<<<")
        self.__update_cbs()
        model.train(self.__epoch_size, self.__dataset, callbacks=self.__callbacks,
                    dataset_sink_mode=self.__do_sink, sink_size=self.__sink_size)
        print(">>>>>>>>>>>>>>>>>>>>> train success ~ <<<<<<<<<<<<<<<<<<<<<<")

    def __eval(self):
        # import
        from mindspore import Model, load_param_into_net, load_checkpoint
        from mindspore.nn.metrics import Accuracy

        # load params
        if self.__ckpt_path:
            load_param_into_net(self.__network, load_checkpoint(self.__ckpt_path))
        else:
            print("Warning: `ckpt_path` is None, Please call func: `set_ckpt_path($ckpt_path)`.")
            return

        # loss_fn & optimizer & metrics
        model = Model(self.__network, loss_fn=self.__loss_fn, optimizer=self.__optimizer,
                      metrics={"Accuracy": Accuracy()} if self.__metrics is None else self.__metrics)

        # eval
        print(">>>>>>>>>>>>>>>>>>>>> eval start ... <<<<<<<<<<<<<<<<<<<<<<")
        result = model.eval(self.__dataset)
        print(">>>>>>>>>>>>>>>>>>>>> eval success ~ <<<<<<<<<<<<<<<<<<<<<<: result=", result)

    def __run_standalone(self):
        # import
        from mindspore import context
        from mindspore.communication import init
        from mindspore.context import ParallelMode

        # set context: device_target
        context.set_context(device_target=self.__device_target)

        # set context: mode
        if self.__graph_mode:
            context.set_context(mode=context.GRAPH_MODE)

        # set context: save_graphs
        context.set_context(save_graphs=self.__save_graphs)

        # set context: device_id
        device_id = int(os.environ.get("DEVICE_ID", 0))
        context.set_context(device_id=device_id)

        # init
        device_num = int(os.environ.get("DEVICE_NUM", 1))
        if device_num > 1 and "win32" not in sys.platform:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()

        if self.__dataset is None:
            print("Warning: `dataset` is None. Please call func: `set_dataset($dataset)`.")

        if self.__network is None:
            print("Warning: `network` is None. Please call func: `set_network($network)`.")

        if self.__dataset is None or self.__network is None:
            return

        if self.__do_eval:
            self.__eval()
        else:
            self.__train()

    def __run_distribute(self, vds):
        train_name = '_'.join(['output', self.__name])
        net_output_dir = os.path.join(os.getcwd(), train_name)
        net_output_dir = eut.create_dir(net_output_dir)
        script_file = os.path.abspath(sys.argv[0])
        script_args = " ".join([x for x in sys.argv[1:] if "--visible_devices" not in x and "-vd" not in x])
        script_args = script_args + " no_distribute"

        rank_table_file = gen_rank_table_file(visible_devices=vds)
        rank_size = len(vds)
        log_file = None
        for rank_id in range(rank_size):
            device_id = vds[rank_id]
            cmd = list()
            cmd.append("export RANK_SIZE={}".format(rank_size))
            cmd.append("export RANK_ID={}".format(rank_id))
            cmd.append("export DEVICE_NUM={}".format(rank_size))
            cmd.append("export DEVICE_ID={}".format(device_id))
            cmd.append("export RANK_TABLE_FILE={}".format(rank_table_file))
            cmd.append("export MINDSPORE_HCCL_CONFIG_PATH={}".format(rank_table_file))

            device_run_dir = os.path.join(net_output_dir, "device{}".format(device_id))
            log_file = os.path.join(device_run_dir, "out.log")
            python_cmd = "python {} {} > {} 2>&1".format(script_file, script_args, log_file)

            cmd.append("rm -rf {}".format(device_run_dir))
            cmd.append("mkdir {}".format(device_run_dir))
            cmd.append("cd {}".format(device_run_dir))
            cmd.append("{} &".format(python_cmd))

            if "win32" in sys.platform:
                cmd = list()
                cmd.append("set RANK_SIZE={}".format(rank_size))
                cmd.append("set RANK_ID={}".format(rank_id))
                cmd.append("set DEVICE_NUM={}".format(rank_size))
                cmd.append("set DEVICE_ID={}".format(device_id))
                device_run_dir = device_run_dir.replace('/', '\\')
                if os.path.exists(device_run_dir):
                    cmd.append("rd /s /q {}".format(device_run_dir))
                cmd.append("mkdir {}".format(device_run_dir))
                cmd.append("cd {}".format(device_run_dir))
                cmd.append("start /b {}".format(python_cmd))

            run_cmd = " && ".join(cmd)
            # print("Run cmd: {}".format(run_cmd))
            os.system(run_cmd)
            print("Output log: {}".format(log_file))

        if log_file and os.path.exists(log_file):
            import threading
            thread = threading.Thread(target=lambda: os.system("tail -f {}".format(log_file)))
            thread.daemon = True
            thread.start()

    def run(self):
        self.__parser.print_args()
        if self.__visible_devices and 'no_distribute' not in sys.argv:
            vds = self.__visible_devices.split(",")
            self.__run_distribute(vds)
        else:
            self.__run_standalone()
