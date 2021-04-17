import click

import erhsh
from erhsh.ms.demo import demo_mgmt


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("Supported MindSpore Demo version is %s" % erhsh.ms.demo.version.__version__)
    ctx.exit()


@click.group(help='This is MindSpore Demo Command Line Console.')
@click.option("--version", is_flag=True, callback=print_version, expose_value=False,
              help="Show Supported MindSpore Demo version.")
def msd_cli():
    pass


@msd_cli.command("ds", help="MindSpore Dataset Fetcher")
@click.argument("name", type=str, default="")
@click.option("--fetch_to", "-ft", type=str, help="fetch to DIR.")
@click.option("--view", "-v", is_flag=True, default=False, help="view in screen")
def ds(name, fetch_to, view):
    mgmt = demo_mgmt.DemoMgmt(name="ds", dir_name="dataset")
    if name:
        mgmt.fetch_demo(name, fetch_to, view)
    else:
        mgmt.print_demo_list()


@msd_cli.command("net", help="MindSpore Network Fetcher")
@click.argument("name", type=str, default="")
@click.option("--fetch_to", "-ft", type=str, help="fetch to DIR.")
@click.option("--view", "-v", is_flag=True, default=False, help="view in screen")
def net(name, fetch_to, view):
    mgmt = demo_mgmt.DemoMgmt(name="net", dir_name="network")
    if name:
        mgmt.fetch_demo(name, fetch_to, view)
    else:
        mgmt.print_demo_list()


@msd_cli.command("ops", help="MindSpore Operator Fetcher")
@click.argument("name", type=str, default="")
@click.option("--fetch_to", "-ft", type=str, help="fetch to DIR.")
@click.option("--view", "-v", is_flag=True, default=False, help="view in screen")
def ops(name, fetch_to, view):
    mgmt = demo_mgmt.DemoMgmt(name="ops", dir_name="operator")
    if name:
        mgmt.fetch_demo(name, fetch_to, view)
    else:
        mgmt.print_demo_list()


@msd_cli.command("res", help="MindSpore Resource Fetcher")
@click.argument("name", type=str, default="")
@click.option("--fetch_to", "-ft", type=str, help="fetch to DIR.")
@click.option("--view", "-v", is_flag=True, default=False, help="view in screen")
def res(name, fetch_to, view):
    mgmt = demo_mgmt.DemoMgmt(name="res", dir_name="resource")
    if name:
        mgmt.fetch_demo(name, fetch_to, view)
    else:
        mgmt.print_demo_list()


@msd_cli.command("train", help="MindSpore Training Fetcher")
@click.argument("name", type=str, default="")
@click.option("--fetch_to", "-ft", type=str, help="fetch to DIR.")
@click.option("--view", "-v", is_flag=True, default=False, help="view in screen")
def train(name, fetch_to, view):
    mgmt = demo_mgmt.DemoMgmt(name="train", dir_name="training")
    if name:
        mgmt.fetch_demo(name, fetch_to, view)
    else:
        mgmt.print_demo_list()


if __name__ == '__main__':
    msd_cli()
