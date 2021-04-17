import click

import erhsh
from erhsh.pt.demo import demo_mgmt


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("Supported PyTorch Demo version is %s" % erhsh.pt.demo.version.__version__)
    ctx.exit()


@click.group(help='This is PyTorch Demo Command Line Console.')
@click.option("--version", is_flag=True, callback=print_version, expose_value=False,
              help="Show Supported PyTorch Demo version.")
def ptd_cli():
    pass


@ptd_cli.command("ops", help="PyTorch Operator Fetcher")
@click.argument("name", type=str, default="")
@click.option("--fetch_to", "-ft", type=str, help="fetch to DIR.")
@click.option("--view", "-v", is_flag=True, default=False, help="view in screen")
def ops(name, fetch_to, view):
    mgmt = demo_mgmt.DemoMgmt(name="ops", dir_name="operator")
    if name:
        mgmt.fetch_demo(name, fetch_to, view)
    else:
        mgmt.print_demo_list()


if __name__ == '__main__':
    ptd_cli()
