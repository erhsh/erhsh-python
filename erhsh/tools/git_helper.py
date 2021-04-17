from erhsh.utils import TblPrinter


class GitHelper:
    def __init__(self):
        self.cmd_items = list()

    def add(self, *cmd_item):
        self.cmd_items.append(cmd_item)

    def show(self):
        tp = TblPrinter("Name", "Cmd Example")
        for cmd_item in self.cmd_items:
            tp.add_row(*cmd_item)
        tp.print()


git_helper = GitHelper()

git_helper.add('git lg', '''
    git config --global alias.lg "log --pretty=format:'%C(yellow)%h%Creset - [%Cgreen%ci %C(bold blue)%>(16,trunc)%cn%Creset] %<(80,trunc)%s %Cgreen(%cr) %C(cyan)%d%Creset'"
'''.strip())
git_helper.add('git blame <file>', 'git blame test.py -L 10')
git_helper.add('git log -- <file>', 'git log -- test.py')
git_helper.add('git log -p -- <file>', 'git log -p -2 --test.py')
git_helper.add('git sslVerify', 'git config --global http.sslVerify false')
git_helper.add('git postBuffer', 'git config --global http.postBuffer 1048576000')
