from erhsh.utils import TblPrinter


class JupyterHelper:
    def __init__(self):
        self.cmd_items = list()

    def add(self, *cmd_item):
        self.cmd_items.append(cmd_item)

    def show(self):
        tp = TblPrinter("Name", "Cmd Example")
        for cmd_item in self.cmd_items:
            tp.add_row(*cmd_item)
        tp.print()


jupyter_helper = JupyterHelper()

jupyter_helper.add("*install", '''
pip install jupyter
'''.strip())
jupyter_helper.add("*generate config", '''
jupyter notebook --generate-config
'''.strip())
jupyter_helper.add("*password", '''
jupyter notebook password
'''.strip())
jupyter_helper.add("edit config", '''
c.NotebookApp.ip="*"; c.NotebookApp.open_browser=False; c.NotebookApp.notebook_dir="/home/workspace/";
'''.strip())
jupyter_helper.add("contrib nbextensions", '''
pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install --user
'''.strip())
jupyter_helper.add("nbextensions", '''
pip install jupyter_nbextensions_configurator && jupyter nbextensions_configurator enable --user
'''.strip())
jupyter_helper.add("*start", '''
jupyter notebook --notebook-dir='./' --ip='*' --port=8888 --allow-root --no-browser > out.log 2>&1 &
'''.strip())
