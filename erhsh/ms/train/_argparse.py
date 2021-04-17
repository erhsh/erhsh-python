import argparse
import erhsh.utils as eut


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument("--device_target", type=str, default="CPU", choices=['CPU', 'Ascend', 'GPU'],
                          help="device target. (Default: CPU)")
        self.add_argument("--graph_mode", action="store_true", help="graph mode flag. (Default: False)")
        self.add_argument("--save_graphs", action="store_true", help="save graphs flag. (Default: False)")
        self.add_argument("--visible_devices", type=str, help="num0,num1,num2...")
        self.add_argument("--do_eval", action="store_true", help="do eval flag. (Default: False)")
        self.add_argument("--ckpt_path", type=str, default="./ckpt", help="save or eval ckpt path. (Default: ./ckpt)")
        self.add_argument("--epoch_size", type=int, default=1, help="epoch size. (Default: 1)")
        self.add_argument("--no_sink", action="store_false", help="sink flag. (Default: True)")
        self.add_argument("--sink_size", type=int, default=-1, help="sink size. (Default: -1)")
        self.add_argument("--data_path", type=str, default="./data", help="data path. (Default: ./data)")
        self.add_argument("--batch_size", type=int, default=32, help="batch size. (Default: 32)")

        self._args = None
        self._unparsed = None

    def print_args(self):
        tbl = eut.TblPrinter("Param Key", "Param Value")
        tbl.add_row("--device_target", str(self.device_target))
        tbl.add_row("--graph_mode", str(self.graph_mode))
        tbl.add_row("--save_graphs", str(self.save_graphs))
        tbl.add_row("--visible_devices", str(self.visible_devices))
        tbl.add_row("--do_eval", str(self.do_eval))
        tbl.add_row("--ckpt_path", str(self.ckpt_path))
        tbl.add_row("--epoch_size", str(self.epoch_size))
        tbl.add_row("--do_sink", str(self.do_sink))
        tbl.add_row("--sink_size", str(self.sink_size))
        tbl.add_row("--data_path", str(self.data_path))
        tbl.add_row("--batch_size", str(self.batch_size))
        tbl.print()

    def parse_known_args(self, args_=None, namespace=None):
        if self._args:
            return self._args, self._unparsed

        self._args, self._unparsed = super(ArgumentParser, self).parse_known_args(args=args_, namespace=namespace)
        return self._args, self._unparsed

    # device_target
    @property
    def device_target(self):
        return self._args.device_target

    @device_target.setter
    def device_target(self, val):
        self._args.device_target = val

    # graph_mode
    @property
    def graph_mode(self):
        return self._args.graph_mode

    @graph_mode.setter
    def graph_mode(self, val):
        self._args.graph_mode = val

    # save_graphs
    @property
    def save_graphs(self):
        return self._args.save_graphs

    @save_graphs.setter
    def save_graphs(self, val):
        self._args.save_graphs = val

    # visible_devices
    @property
    def visible_devices(self):
        return self._args.visible_devices

    @visible_devices.setter
    def visible_devices(self, val):
        self._args.visible_devices = val

    # do_eval
    @property
    def do_eval(self):
        return self._args.do_eval

    @do_eval.setter
    def do_eval(self, val):
        self._args.do_eval = val

    # ckpt_path
    @property
    def ckpt_path(self):
        return self._args.ckpt_path

    @ckpt_path.setter
    def ckpt_path(self, val):
        self._args.ckpt_path = val

    # epoch_size
    @property
    def epoch_size(self):
        return self._args.epoch_size

    @epoch_size.setter
    def epoch_size(self, val):
        self._args.epoch_size = val

    # do_sink
    @property
    def do_sink(self):
        return self._args.no_sink

    @do_sink.setter
    def do_sink(self, val):
        self._args.no_sink = val

    # sink_size
    @property
    def sink_size(self):
        return self._args.sink_size

    @sink_size.setter
    def sink_size(self, val):
        self._args.sink_size = val

    # data_path
    @property
    def data_path(self):
        return self._args.data_path

    @data_path.setter
    def data_path(self, val):
        self._args.data_path = val

    # batch_size
    @property
    def batch_size(self):
        return self._args.batch_size

    @batch_size.setter
    def batch_size(self, val):
        self._args.batch_size = val

    def __getattr__(self, item):
        if hasattr(self._args, item):
            return getattr(self._args, item)
        return None

    def __repr__(self):
        return str(self._args)

    def __str__(self):
        return str(self._args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="dataset path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    args = parser.parse_known_args()
    args.data_path = "/my/data/path"
    args.epoch_size = 123
    args.do_sink = False

    print(type(args))
    print(args)
    print(args.data_path)
    print(args.epoch_size)
    print(args.do_sink)
