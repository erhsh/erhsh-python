from erhsh.utils import TblPrinter


class VimHelper:
    def __init__(self):
        self.cmd_items = list()

    def add(self, *cmd_item):
        self.cmd_items.append(cmd_item)

    def show(self):
        tp = TblPrinter("Name", "Cmd Example")
        for cmd_item in self.cmd_items:
            tp.add_row(*cmd_item)
        tp.print()


vim_helper = VimHelper()

vim_helper.add('show unicode', ':digraphs')
vim_helper.add('visualize chars', '1. :set list     2. :set listchars=eol:$,tab:>-,trail:~,extends:>,precedes:<')
vim_helper.add('tab2space', '1. :set tabstop=4  2. :set shiftwidth=4    3. :set expandtab')
