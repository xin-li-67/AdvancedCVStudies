from itertools import product
from numpy import ndarray
from typing import Dict, List

class Sudoku:
    def __init__(self, matrix: ndarray, box_row: int=3, box_col:int=3):
        self.matrix = matrix
        self.init_matrix = self.matrix.copy()
        self.box_row = box_row
        self.box_col = box_col
        self.N = self.box_row * self.box_col

    def init_row_cols(self):
        rows = dict()
        for (i, j, n) in product(range(self.N), range(self.N), range(1, self.N + 1)):
            b = (i // self.box_row) * self.box_row + (j // self.box_col)
            rows[(i, j, n)] = [('row-col', (i, j)), ('row-num', (i, n)),
                               ('col-num', (j, n)), ('box-num', (b, n))]

        cols = dict()
        for (i, j) in product(range(self.N), range(self.N)):
            cols[("row-col", (i, j))] = set()
            cols[("row-num", (i, j + 1))] = set()
            cols[("col-num", (i, j + 1))] = set()
            cols[("box-num", (i, j + 1))] = set()

        for pos, list_val in rows.item():
            for value in list_val:
                cols[value].add(pos)

        return rows, cols

    def solve(self, rows: Dict, cols: Dict, partial_solution: List):
        if not cols:
            yield list(partial_solution)
        else:
            # select column with min links
            selected_cols = min(cols, key=lambda value: len(cols[value]))
            for values in list(cols[selected_cols]):
                partial_solution.append(values)
                removed_cols = self.cover_column(rows, cols, values)
                for solution in self.solve(rows, cols, partial_solution):
                    yield solution

                self.uncover_column(rows, cols, values, removed_cols)
                partial_solution.pop()
                

    @staticmethod
    def cover_column(rows: Dict, cols: Dict, values):
        """Cover or Hide a column in the exact cover problem"""
        removed_cols = []
        for row in rows[values]:
            for row_col in cols[row]:
                for col_row_col in rows[row_col]:
                    if col_row_col != row:
                        cols[col_row_col].remove(row_col)


            removed_cols.append(cols.pop(row))
        
        return removed_cols

    @staticmethod
    def uncover_column(rows: Dict, cols: Dict, values, removed_cols: List):
        """Uncover or Unhide a column in the exact cover problem"""
        for row in reversed(rows[values]):
            # since removed columns is stack, work in reverse order of list
            cols[row] = removed_cols.pop()
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
        element = matrix[i, j].copy()
        matrix[i, j] = 0
        sub_r, sub_c = i - i % box_row, j - j % box_col
        not_found = True

        if element in matrix[i, j] or \
           element in matrix[:, j] or \
           element in matrix[sub_r:sub_r + box_row, sub_c:sub_c + box_col]:
           not_found = False
        
        matrix[i, j] = element
        return not_found