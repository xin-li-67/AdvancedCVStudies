from os.path import exists

import pygame
import numpy as np

from game_window import SudokuGUI

pygame.init()
pygame.display.set_caption('AR Sudoku')
icon = pygame.image.load('./.icons/game_logo.png')
pygame.display.set_icon(icon)


def main_game():
    while True:
        if exists('last_loaded.npy') and exists('last_loaded_dim.npy'):
            mat = np.load('last_loaded.npy')
            box_rows, box_cols = tuple(np.load('last_loaded_dim.npy'))
        else:
            mat = np.load('sample.npy')
            box_rows, box_cols = (3, 3)

        sg = SudokuGUI(mat.copy(), box_rows, box_cols)

        menu_val = 0
        while not menu_val:
            menu_val = sg.main_menu()

        if menu_val == 1:
            while sg.play_game():
                pass
        else:
            while sg.load_AR():
                pass


if __name__ == '__main__':
    main_game()
