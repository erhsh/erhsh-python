import time


def get_time_monitor():
    from mindspore.train.callback import Callback

    class _TimeMonitor(Callback):
        def __init__(self):
            super(_TimeMonitor, self).__init__()
            self.epoch_start_time = 0
            self.epoch_end_time = 0
            self.epoch_start_step_num = 0
            self.epoch_end_step_num = 0

        def epoch_begin(self, run_context):
            self.epoch_start_time = time.time()

            cb_params = run_context.original_args()
            self.epoch_start_step_num = cb_params['cur_step_num']

        def epoch_end(self, run_context):
            self.epoch_end_time = time.time()

            cb_params = run_context.original_args()
            self.epoch_end_step_num = cb_params['cur_step_num']

            epoch_num = cb_params['epoch_num']
            cur_epoch_num = cb_params['cur_epoch_num']

            epoch_time_cost = (self.epoch_end_time - self.epoch_start_time) * 1000
            epoch_step_num = self.epoch_end_step_num - self.epoch_start_step_num
            per_step_cost = epoch_time_cost / epoch_step_num

            msg = "Epoch[{:2d}/{:2d}], Step[{:2d}], epoch time: {:5.3f}ms, per step time: {:5.3f}ms". \
                format(cur_epoch_num, epoch_num, self.epoch_end_step_num, epoch_time_cost, per_step_cost)

            print(msg, flush=True)

    return _TimeMonitor()


TimeMonitor = get_time_monitor
