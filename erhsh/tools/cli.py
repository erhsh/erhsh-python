import click

import erhsh
from erhsh.tools.bash_helper import bash_helper
from erhsh.tools.git_helper import git_helper
from erhsh.tools.jupyter_helper import jupyter_helper
from erhsh.tools.vim_helper import vim_helper


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


if __name__ == '__main__':
    tools_cli()
