import time

import numpy as np

np.set_printoptions(suppress=True)


def print_ds_info(ds, prefix="|-"):
    print("------ Dataset Information Is: ------", flush=True)
    print("{}get_dataset_size={}".format(prefix, ds.get_dataset_size()), flush=True)
    print("{}get_repeate_count={}".format(prefix, ds.get_repeat_count()), flush=True)
    print("{}output_types={}".format(prefix, ds.output_types()), flush=True)
    print("{}output_shapes={}".format(prefix, ds.output_shapes()), flush=True)

    for x in ds.create_dict_iterator():
        keys = x.keys()
        for k in keys:
            v = x.get(k)
            print("{} {} -> {} {}".format(prefix, k, v.shape, v.dtype), flush=True)
        break


def _print_cost(costs, decimals=None, prefix=""):
    print("{}AVG:{}".format(prefix, np.around(np.mean(costs), decimals=decimals)), flush=True)
    print("{}99%:{}".format(prefix, np.around(np.percentile(costs, 99), decimals=decimals)), flush=True)
    print("{}90%:{}".format(prefix, np.around(np.percentile(costs, 90), decimals=decimals)), flush=True)
    print("{}75%:{}".format(prefix, np.around(np.percentile(costs, 75), decimals=decimals)), flush=True)
    print("{}50%:{}".format(prefix, np.around(np.percentile(costs, 50), decimals=decimals)), flush=True)
    print("{}Max:{}".format(prefix, np.around(np.max(costs), decimals=decimals)), flush=True)
    print("{}Min:{}".format(prefix, np.around(np.min(costs), decimals=decimals)), flush=True)
    print("{}Sum:{}".format(prefix, np.around(np.sum(costs), decimals=decimals)), flush=True)


def print_ds_performance(ds, step1=10, step2=None, decimals=9, prefix="|-"):
    i = 0
    costs = []
    ss_time = s_time = time.time()
    for _ in ds.create_dict_iterator():
        i += 1
        e_time = time.time()
        c_time = round(e_time - s_time, decimals)
        s_time = e_time

        costs.append(c_time)

        if step2 and i % step2 == 0:
            print("cur index:", i, ", cur cost:", c_time, flush=True)
            _print_cost(costs, decimals=decimals, prefix=prefix)
        elif step1 and i % step1 == 0:
            print("cur index:", i, ", cur cost:", c_time, flush=True)

    print("Total Count: {}, total cost: {}".format(i, round(time.time() - ss_time, decimals)), flush=True)
    _print_cost(costs, decimals=decimals, prefix=prefix)

    print("Save numpy start...")
    np.savetxt("time_costs.txt", np.array(costs), fmt="%.{}f".format(decimals))
    print("Save numpy success~")


def _get_value(v):
    from mindspore import Tensor
    if isinstance(v, Tensor):
        v = v.asnumpy()

    return v


def print_ds_data(ds, prefix=">>>", row_limit=10, col_limit=100):
    i = 0
    for x in ds.create_dict_iterator():
        i += 1
        if i > row_limit:
            break
        print("-" * 32, "item:", i, "-" * 32, flush=True)
        keys = x.keys()
        for k in keys:
            v = x.get(k)
            vv = _get_value(v).flatten()
            if len(vv) > col_limit:
                limit = int(col_limit / 2)
                print("{} {}->{}\n{}\n...\n{}".format(prefix, k, v.shape, vv[:limit], vv[-limit:]), flush=True)
            else:
                print("{} {}->{}\n{}".format(prefix, k, v.shape, vv[:]), flush=True)
