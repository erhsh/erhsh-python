import click

import erhsh
from erhsh.tools.bash_helper import bash_helper
from erhsh.tools.git_helper import git_helper
from erhsh.tools.jupyter_helper import jupyter_helper
from erhsh.tools.vim_helper import vim_helper

from erhsh import utils as eut


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("tools version is %s" % erhsh.tools.version.__version__)
    ctx.exit()


@click.group(help='This is Tools Command Line Console.')
@click.option("--version", is_flag=True, callback=print_version, expose_value=False,
              help="Show Tools Command Line Version.")
def tools_cli():
    pass


@tools_cli.command("git", help="Get Git Helper")
def git():
    git_helper.show()


@tools_cli.command("vim", help="Get Vim Helper")
def vim():
    vim_helper.show()


@tools_cli.command("bash", help="Get Bash Helper")
def bash():
    bash_helper.show()


@tools_cli.command("jupyter", help="Get Jupyter Helper")
def jupyter():
    jupyter_helper.show()


@tools_cli.command("gray2rgb", help="Conver Gray Image to RGB")
@click.option("--src_path", "-s", required=True, help="src gray image file path")
@click.option("--dest_path", "-d", default="./", help="src gray image file path")
@click.option("--h_num", "-h", default=1, help="src gray image file path")
@click.option("--w_num", "-w", default=1, help="src gray image file path")
def gray2rgb(src_path, dest_path, h_num, w_num):
    if h_num > 1 or w_num > 1:
        eut.convertGray2RGB_Muti(src_path, dest_path=dest_path, h_num=h_num, w_num=w_num)
    else:
        eut.convertGray2RGB(src_path, dest_path=dest_path)


if __name__ == '__main__':
    tools_cli()
