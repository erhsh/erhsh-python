import json
import os

DEVICE_NUM = 8
HCCN_CONF_FILE = '/etc/hccn.conf'
BOARD_IDS = {
    'A+K': '0x002f',
    'A+X': '0x0000'
}
DEVICE_TYPE = 'A+K'

SERVER_ID = '10.10.10.10'


def gen_rank_table_file(server_id=SERVER_ID, device_type=DEVICE_TYPE, visible_devices=None,
                        hccn_conf_file=None):
    server_id = SERVER_ID if server_id is None else server_id
    device_type = DEVICE_TYPE if device_type is None else device_type
    visible_devices = [str(x) for x in range(DEVICE_NUM)] if visible_devices is None else visible_devices
    hccn_conf_file = HCCN_CONF_FILE if hccn_conf_file is None else hccn_conf_file

    device_num = len(visible_devices)

    def get_device_ips(cfg_file):
        rst = {}
        with open(cfg_file, 'r') as f:
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
            '0': '192.168.1.101',
            '1': '192.168.2.101',
        }
        device_num = len(device_ips)
        visible_devices = [str(x) for x in range(device_num)]

    # rank_table template
    rank_table = {
        'board_id': BOARD_IDS.get(device_type),
        'chip_info': '910',
        'deploy_mode': 'lab',
        'group_count': '1',
        'group_list': [],  # set later
        'para_plane_nic_location': 'device',
        'para_plane_nic_num': '',
        'para_plane_nic_name': [],  # set later
        'status': 'completed'
    }
    group_list = [
        {
            'device_num': str(device_num),
            'server_num': '1',
            'group_name': '',
            'instance_count': str(device_num),
            'instance_list': [],  # set later
        }
    ]
    rank_table['group_list'] = group_list

    instance_list = []
    for i in range(device_num):
        instance = {
            'rank_id': str(i),
            'server_id': server_id,
            'devices': {
                'device_id': visible_devices[i],
                'device_ip': device_ips[visible_devices[i]],
            }
        }
        instance_list.append(instance)
    group_list[0]['instance_list'] = instance_list

    para_plane_nic_name = []
    for i in range(device_num):
        eth_id = visible_devices[i]
        para_plane_nic_name.append('eth{}'.format(eth_id))
    rank_table['para_plane_nic_name'] = para_plane_nic_name

    usable_dev = ''.join(visible_devices)
    table_fn = os.path.join(os.getcwd(), 'rank_table_{}p_{}_{}.json'.format(device_num, usable_dev, server_id))
    with open(table_fn, 'w') as table_fp:
        json.dump(rank_table, table_fp, indent=4)
    return table_fn


if __name__ == '__main__':
    gen_rank_table_file()
