import os

import click

import erhsh
from erhsh.ms.tools.ckpt import MsCkptLoader
from erhsh.ms.tools.bin_tool import BinLoader
from erhsh.ms.tools.hccl_tool import gen_rank_table_file
from erhsh.ms.tools.hccl_tool_v1 import gen_rank_table_file as gen_rank_table_file_v1


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("Supported MindSpore version is %s" % erhsh.ms.version.__version__)
    ctx.exit()


@click.group(help='This is MindSpore Command Line Console.')
@click.option("--version", is_flag=True, callback=print_version, expose_value=False,
              help="Show Supported MindSpore version.")
def ms_cli():
    pass


@ms_cli.command("hccl", help="Generate HCCL Rank Table File")
@click.option("--server_id", "-sid", prompt="Server Id(IP)", help="server id")
@click.option("--device_type", "-dt", prompt="Device Type", help="device type")
@click.option("--visible_devices", "-vds", help="visible devices")
@click.option("--hccn_conf_file", "-hcf", help="hccn config file")
def gen_hccl_rank_table(server_id, device_type, visible_devices, hccn_conf_file):
    gen_file_path = gen_rank_table_file(server_id=server_id, device_type=device_type, visible_devices=visible_devices,
                                        hccn_conf_file=hccn_conf_file)
    print("Generate success:", gen_file_path)


@ms_cli.command("hcclv1", help="Generate HCCL Rank Table File v1.0")
@click.option("--server_id", "-sid", prompt="Server Id(IP)",
              default=lambda: os.getenv("SSH_CONNECTION", "ip port x.x.x.x").split()[2], help="server id")
@click.option("--visible_devices", "-vds", prompt="Visible Devices",
              default="0,1,2,3,4,5,6,7",  help="visible devices")
@click.option("--hccn_conf_file", "-hcf", help="hccn config file")
def gen_hccl_rank_table_v1(server_id, visible_devices, hccn_conf_file):
    gen_file_path = gen_rank_table_file_v1(server_id=server_id, visible_devices=visible_devices,
                                           hccn_conf_file=hccn_conf_file)
    print("Generate success:", gen_file_path)


@ms_cli.command("ckpt", help="Get checkpoint information")
@click.option("--ckpt_path", "-p", required=True, help="checkpoint file path")
@click.option("--filter_key", "-f", default=None, help="key filter condition, regular supported")
@click.option("--key", "-k", default=None, help="specify key")
@click.option("--dump", "-d", is_flag=True, default=False, help="flag dump data")
@click.option("--dump_to", "-t", default="./", help="where dump to")
def ckpt(ckpt_path, filter_key, key, dump, dump_to):
    ms_loader = MsCkptLoader(ckpt_path)
    if key:
        if dump:
            ms_loader.get_dump(key, dump_to=dump_to)
        else:
            ms_loader.get(key)
    else:
        if dump:
            ms_loader.list_dump(filter_key=filter_key, dump_to=dump_to)
        else:
            ms_loader.list(filter_key=filter_key)


@ms_cli.command("bin", help="Get bin information")
@click.option("--bin_path", "-p", required=True, help="bin file path")
def bin(bin_path):
    bin_loader = BinLoader(bin_path).load()
    bin_loader.is_nan_exist()
    bin_loader.print_bin()


if __name__ == '__main__':
    ms_cli()
