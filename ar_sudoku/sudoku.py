from itertools import product
from typing import Dict, List

from numpy import ndarray


class Sudoku:
    def __init__(self, matrix: ndarray, box_row: int = 3, box_col: int = 3):
        self.matrix = matrix
        self.init_matrix = self.matrix.copy()
        self.box_row = box_row
        self.box_col = box_col
        self.N = self.box_row * self.box_col

    def init_row_cols(self):
        rows = dict()
        for (i, j, n) in product(range(self.N), range(self.N), range(1, self.N + 1)):
            b = (i // self.box_row) * self.box_row + (j // self.box_col)
            rows[(i, j, n)] = [
                ("row-col", (i, j)), ("row-num", (i, n)),
                ("col-num", (j, n)), ("box-num", (b, n))
            ]

        columns = dict()
        for (i, j) in product(range(self.N), range(self.N)):
            columns[("row-col", (i, j))] = set()
            columns[("row-num", (i, j + 1))] = set()
            columns[("col-num", (i, j + 1))] = set()
            columns[("box-num", (i, j + 1))] = set()

        for pos, list_value in rows.items():
            for value in list_value:
                columns[value].add(pos)

        return rows, columns

    def solve(self, rows: Dict, cols: Dict, partial_solution: List):
        if not cols:
            yield list(partial_solution)
        else:
            selected_col = min(cols, key=lambda value: len(cols[value]))
            for values in list(cols[selected_col]):
                partial_solution.append(values)
                removed_cols = self.cover_column(rows, cols, values)
                for solution in self.solve(rows, cols, partial_solution):
                    yield solution
                self.uncover_column(rows, cols, values, removed_cols)
                partial_solution.pop()

    @staticmethod
    def cover_column(rows: Dict, cols: Dict, values):
        removed_columns = []
        
        for row in rows[values]:
            for row_col in cols[row]:
                for col_row_col in rows[row_col]:
                    if col_row_col != row:
                        cols[col_row_col].remove(row_col)

            removed_columns.append(cols.pop(row))
        return removed_columns

    @staticmethod
    def uncover_column(rows: Dict, cols: Dict, values, removed_columns: List):
        for row in reversed(rows[values]):
            cols[row] = removed_columns.pop()
            for row_col in cols[row]:
                for col_row_col in rows[row_col]:
                    if col_row_col != row:
                        cols[col_row_col].add(row_col)

    def get_solution(self):
        rows, cols = self.init_row_cols()
        solutions = []

        for i in range(self.N):
            for j in range(self.N):
                if self.matrix[i, j] != 0:
                    self.cover_column(rows, cols, (i, j, self.matrix[i, j]))

        for solution in self.solve(rows, cols, []):
            for (i, j, element) in solution:
                self.matrix[i, j] = element
            solutions.append(self.matrix)
            self.matrix = self.init_matrix.copy()
        
        return solutions

    @staticmethod
    def element_possible(matrix: ndarray, box_row: int, box_col: int, i: int, j: int):
        # backup the element in place
        element = matrix[i, j].copy()
        matrix[i, j] = 0
        sub_r, sub_c = i - i % box_row, j - j % box_col

        not_found = True
        if element in matrix[i, :] or  element in matrix[:, j] or \
            element in matrix[sub_r:sub_r + box_row, sub_c:sub_c + box_col]:
            not_found = False

        matrix[i, j] = element
        return not_found
