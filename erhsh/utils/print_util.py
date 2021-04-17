class TblPrinter(object):
    def __init__(self, *header, tbl_name=None, col_separator=" | ", row_separator="-"):
        self._header = header
        self._col_num = len(header)
        self._col_widths = [len(h) for h in header]
        self._rows = []
        self._tbl_name = tbl_name
        self._col_separator = col_separator
        self._row_separator = row_separator

    def add_row(self, *row):
        if len(row) == self._col_num:
            for i in range(self._col_num):
                old_col_width = self._col_widths[i]
                new_col_width = len(row[i])
                if new_col_width > old_col_width:
                    self._col_widths[i] = new_col_width
        self._rows.append(row)
        return self

    def _print_header(self):
        self.__print_line()
        if self._tbl_name:
            print_info = ['{0:^{ilen}}'.format(self._tbl_name, ilen=self.__get_tbl_width())]
            print(self._col_separator.join(print_info))
            self.__print_line()

        print_info = ['{0:^{ilen}}'.format(self._header[i], ilen=ilen) for i, ilen in enumerate(self._col_widths)]
        print(self._col_separator.join(print_info))
        self.__print_line()

    def _print_body(self):
        for row in self._rows:
            if len(row) == self._col_num:
                print_info = ['{0:<{ilen}}'.format(row[i], ilen=ilen) for i, ilen in enumerate(self._col_widths)]
            else:
                print_info = ['{0:<{ilen}}'.format(str(col), ilen=len(str(col))) for i, col in enumerate(row)]
            print(self._col_separator.join(print_info))
        self.__print_line()

    def __print_line(self):
        print(self._row_separator * self.__get_tbl_width())

    def __get_tbl_width(self):
        return sum(self._col_widths) + len(self._col_separator)

    def print(self):
        self._print_header()
        self._print_body()
