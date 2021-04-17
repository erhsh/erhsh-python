from erhsh.utils import TblPrinter


class BashHelper:
    def __init__(self):
        self.cmd_items = list()

    def add(self, *cmd_item):
        self.cmd_items.append(cmd_item)

    def show(self):
        tp = TblPrinter("Name", "Cmd Example")
        for cmd_item in self.cmd_items:
            tp.add_row(*cmd_item)
        tp.print()


bash_helper = BashHelper()

bash_helper.add("tr env", '''
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -n 1 -I {} cat /proc/{}/environ | tr '\\0' '\\n' | grep DEV | sort | uniq
'''.strip())
bash_helper.add("ms commit id", '''
cat `pip show pip | grep Location | awk '{print $2}'`/mindspore/.commit_id
'''.strip())
bash_helper.add("pgrep", '''
pgrep -P 1 -t `who am i | awk '{print $2}'` -f python
'''.strip())
