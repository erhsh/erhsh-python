import click

import erhsh
from erhsh.pt.tools.pth import PtPthLoader


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("Supported PyTorch version is %s" % erhsh.pt.version.__version__)
    ctx.exit()


@click.group(help='This is PyTorch Command Line Console.')
@click.option("--version", is_flag=True, callback=print_version, expose_value=False,
              help="Show Supported PyTorch version.")
def pt_cli():
    pass


@pt_cli.command("pth", help="Get pth information")
@click.option("--pth_path", "-p", required=True, help="checkpoint file path")
@click.option("--filter_key", "-f", default=None, help="key filter condition, regular supported")
@click.option("--key", "-k", default=None, help="specify key")
@click.option("--dump", "-d", is_flag=True, default=False, help="flag dump data")
@click.option("--dump_to", "-t", default="./", help="where dump to")
def pth(pth_path, filter_key, key, dump, dump_to):
    pth_loader = PtPthLoader(pth_path)
    if key:
        if dump:
            pth_loader.get_dump(key, dump_to=dump_to)
        else:
            pth_loader.get(key)
    else:
        if dump:
            pth_loader.list_dump(filter_key=filter_key, dump_to=dump_to)
        else:
            pth_loader.list(filter_key=filter_key)


if __name__ == '__main__':
    pt_cli()
