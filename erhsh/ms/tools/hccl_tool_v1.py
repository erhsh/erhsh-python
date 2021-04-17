import json
import os

DEVICE_NUM = 8
HCCN_CONF_FILE = '/etc/hccn.conf'

SERVER_ID = os.getenv('SSH_CONNECTION', 'ip port x.x.x.x').split()[2]


def gen_rank_table_file(server_id=None, visible_devices=None, hccn_conf_file=None):
    hccn_conf_file = HCCN_CONF_FILE if hccn_conf_file is None else hccn_conf_file

    def get_device_ips(cfg_file):
        rst = {}
        with open(cfg_file, 'r') as f:
            print("hccn_conf_file:", cfg_file)
            for line in f.readlines():
                if not line.startswith('address_'):
                    continue
                device_id, device_ip = line.strip().split('=')
                device_id = device_id.split('_')[1]
                rst[device_id] = device_ip
                print('device_id:{}, device_ip:{}'.format(device_id, device_ip))
        return rst

    if os.path.exists(hccn_conf_file):
        device_ips = get_device_ips(hccn_conf_file)
    else:
        print("Warning: Please specify the param `hccn_conf_file` so that we can get the right device ip.")
        device_ips = {
            '0': '192.168.101.101',
            '1': '192.168.102.101',
            '2': '192.168.103.101',
            '3': '192.168.104.101',
            '4': '192.168.101.102',
            '5': '192.168.102.102',
            '6': '192.168.103.102',
            '7': '192.168.104.102',
        }

    visible_devices = [str(x) for x in range(DEVICE_NUM)] if visible_devices is None else visible_devices
    visible_devices = visible_devices.split(',') if isinstance(visible_devices, str) else visible_devices
    device_num = len(visible_devices)

    # rank_table template
    rank_table = {
        'version': '1.0',
        'server_count': '1',
        'server_list': [],  # set later
        'status': 'completed'
    }
    server_list = [
        {
            'server_id': str(SERVER_ID if server_id is None else server_id),
            'device': [],  # set later
            'host_nic_ip': 'reserve'
        }
    ]
    rank_table['server_list'] = server_list

    device = []
    for i in range(device_num):
        instance = {
            'rank_id': str(i),
            'device_id': visible_devices[i],
            'device_ip': device_ips[visible_devices[i]],
        }
        device.append(instance)
    server_list[0]['device'] = device

    rank_table_file = os.path.join(os.getcwd(), 'rank_table_{}p_{}.json'.format(device_num, '-'.join(visible_devices)))
    with open(rank_table_file, 'w') as table_fp:
        json.dump(rank_table, table_fp, indent=4)

    soft_link_file = 'rank_table_{}p.json'.format(device_num)
    if os.path.exists(soft_link_file):
        os.remove(soft_link_file)
    os.symlink(rank_table_file, soft_link_file)

    return rank_table_file


if __name__ == '__main__':
    gen_rank_table_file()
