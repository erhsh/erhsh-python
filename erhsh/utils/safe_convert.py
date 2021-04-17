def safe2int(val, def_val=None):
    try:
        return int(val)
    except ValueError:
        return val if def_val is None else def_val


def safe2float(val, def_val=None):
    try:
        return float(val)
    except ValueError:
        return val if def_val is None else def_val
