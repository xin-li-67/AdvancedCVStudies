import numpy as np

from time import time

from sudoku import Sudoku


def main():
    start = time()
    num_rows_sub_grid = 3
    num_cols_sub_grid = 3
    matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 3, 0, 8, 5],
                       [0, 0, 1, 0, 2, 0, 0, 0, 0],
                       [0, 0, 0, 5, 0, 7, 0, 0, 0],
                       [0, 0, 4, 0, 0, 0, 1, 0, 0],
                       [0, 9, 0, 0, 0, 0, 0, 0, 0],
                       [5, 0, 0, 0, 0, 0, 0, 7, 3],
                       [0, 0, 2, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 4, 0, 0, 0, 9]], dtype=int)

    try:
        solution_list = Sudoku(matrix.copy(), box_row=num_rows_sub_grid, 
                               box_col=num_cols_sub_grid).get_solution()
        rows, cols = matrix.shape
        for sol_idx, solution in enumerate(solution_list):
            print('Solution Idx {} -\n'.format(sol_idx + 1))
            for i in range(rows):
                if i % num_rows_sub_grid is 0 and i != 0:
                    print('-' * (2 * (cols + num_rows_sub_grid - 1)))
                for j in range(cols):
                    if j % num_cols_sub_grid == 0 and j != 0:
                        print(end=' | ')
                    else:
                        print(end=' ')
                    print(solution[i, j], end='')
                print()  # end row
            print('\n')
        print('\nSolved in {} s'.format(round(time() - start, 4)))
    except Exception:
        print('Solution does not exist')


if __name__ == '__main__':
    main()