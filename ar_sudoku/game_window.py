import os
import pygame
import tkinter.messagebox

import numpy as np

from time import sleep
from tkinter import Tk
from random import choice
from typing import List, Tuple
from contextlib import suppress
from tkinter.filedialog import askopenfilename

from sudoku import Sudoku
from gui.button import Button
from gui.camera_window import CameraWindow
from image_processing.process_image import SudokuImageProcessing

os.environ['SDL_VIDEO_CENTERED'] = '1'

root = Tk()
root.withdraw()
BLOCK_SIZE = 40
SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
X = 0
Y = 1


class SudokuGUI:
    BOX_ROWS: int
    BOX_COLS: int
    NUM_ROWS: int
    NUM_COLUMNS: int
    PLAY_WIDTH: int
    PLAY_HEIGHT: int
    TOP_LEFT: Tuple[int, int]
    matrix: np.ndarray
    init_matrix: np.ndarray
    solution_list: List[np.ndarray]
    solution: np.ndarray
    window: pygame.Surface
    selected_box: Tuple[int, int]
    locked_pos: List[Tuple[int, int]]
    home_icon: pygame.Surface
    button_home: Button
    button_load_image: Button
    button_cam_image: Button
    button_solve: Button
    button_play_game: Button
    button_AR: Button

    def __init__(self, matrix: np.ndarray, box_rows: int = 3, box_cols: int = 3):
        self.BOX_ROWS = box_rows
        self.BOX_COLS = box_cols
        self.NUM_ROWS = self.BOX_ROWS * self.BOX_COLS
        self.NUM_COLUMNS = self.BOX_ROWS * self.BOX_COLS
        self.PLAY_WIDTH = BLOCK_SIZE * self.NUM_COLUMNS
        self.PLAY_HEIGHT = BLOCK_SIZE * self.NUM_ROWS
        self.TOP_LEFT = (
            int((SCREEN_WIDTH - self.PLAY_WIDTH) / 2),
            int((SCREEN_HEIGHT - self.PLAY_HEIGHT) / 2 - 80)
        )

        self.matrix = matrix
        self.init_matrix = self.matrix.copy()
        
        try:
            self.solution_list = Sudoku(matrix.copy(), box_row=self.BOX_ROWS, box_col=self.BOX_COLS).get_solution()
            self.solution = self.solution_list[0]
        except Exception:
            tkinter.messagebox.showerror(title="Error",
                                         message="Solution does not exist or Image not clear, Try Again.")
        
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.selected_box = (0, 0)
        self.locked_pos = self.get_locked_pos()
        self.home_icon = pygame.image.load('.images/home_icon.png')
        self.button_home = Button(60, 60, 70, 70, (200,200,200), '  ')
        self.button_load_image = Button(162, 510, 250, 60, (200,200,200), "Load from File")
        self.button_cam_image = Button(488, 510, 250, 60, (200,200,200), "Load from Camera")
        self.button_solve = Button(325, 590, 250, 60, (200,200,200), "Solve")
        self.button_play_game = Button(325, 300, 400, 80, (200,200,200), "Play Game")
        self.button_AR = Button(325, 450, 400, 80, (200,200,200), "Augmented Reality")

    def get_locked_pos(self):
        locked_pos = []
        
        for i in range(self.NUM_ROWS):
            for j in range(self.NUM_COLUMNS):
                if self.matrix[i, j] != 0:
                    locked_pos.append((i, j))
        
        return locked_pos

    def draw_window(self, solved: bool = False):
        self.window.fill((255, 255, 255))
        font = pygame.font.SysFont('comicsans', 48)
        label = font.render('SUDOKU', 1, (0, 0, 0))
        self.window.blit(label, (self.TOP_LEFT[X] + self.PLAY_WIDTH / 2 - (label.get_width() / 2),
                         40 - (label.get_height() / 2)))

        # draw reference grid black lines
        for i in range(self.NUM_ROWS):
            # horizontal lines
            pygame.draw.line(self.window, (0,0,0),
                             (self.TOP_LEFT[X], self.TOP_LEFT[Y] + i * BLOCK_SIZE),
                             (self.TOP_LEFT[X] + self.PLAY_WIDTH, self.TOP_LEFT[Y] + i * BLOCK_SIZE),
                             4 if i % self.BOX_ROWS == 0 else 1)
        for i in range(self.NUM_COLUMNS):
            # vertical lines
            pygame.draw.line(self.window, (0,0,0),
                             (self.TOP_LEFT[X] + i * BLOCK_SIZE, self.TOP_LEFT[Y]),
                             (self.TOP_LEFT[X] + i * BLOCK_SIZE, self.TOP_LEFT[Y] + self.PLAY_HEIGHT),
                             4 if i % self.BOX_COLS == 0 else 1)

        # last horizontal line
        pygame.draw.line(self.window, (0,0,0),
                         (self.TOP_LEFT[X], self.TOP_LEFT[Y] + self.NUM_ROWS * BLOCK_SIZE),
                         (self.TOP_LEFT[X] + self.PLAY_WIDTH, self.TOP_LEFT[Y] + self.NUM_ROWS * BLOCK_SIZE), 4)
        # last vertical line
        pygame.draw.line(self.window, (0,0,0),
                         (self.TOP_LEFT[X] + self.NUM_COLUMNS * BLOCK_SIZE, self.TOP_LEFT[Y]),
                         (self.TOP_LEFT[X] + self.NUM_COLUMNS * BLOCK_SIZE, self.TOP_LEFT[Y] + self.PLAY_HEIGHT), 4)

        # font for the numbers, with different size
        font = pygame.font.SysFont('comicsans', 32)

        for i in range(self.NUM_ROWS):
            for j in range(self.NUM_COLUMNS):
                if self.matrix[i, j] == 0:
                    continue

                if (i, j) in self.locked_pos:
                    num_color = (0, 0, 0)
                elif solved:
                    num_color = (128, 193, 42)
                elif Sudoku.element_possible(self.matrix, self.BOX_ROWS, self.BOX_COLS, i, j):
                    num_color = (89, 154, 252)
                else:
                    num_color = (255, 0, 0)

                label = font.render(str(self.matrix[i, j]), 1, num_color)
                self.window.blit(label,
                                 (self.TOP_LEFT[X] + j * BLOCK_SIZE - label.get_width() / 2 + BLOCK_SIZE / 2,
                                  self.TOP_LEFT[Y] + i * BLOCK_SIZE - label.get_height() / 2 + BLOCK_SIZE / 2))

        # higlight border of the selected box
        pygame.draw.rect(self.window, (100,178,255),
                         (self.TOP_LEFT[X] + self.selected_box[0] * BLOCK_SIZE, self.TOP_LEFT[Y] + self.selected_box[1] * BLOCK_SIZE,
                          BLOCK_SIZE, BLOCK_SIZE), 4)

        self.button_home.draw(self.window)
        self.window.blit(self.home_icon,
                         (self.button_home.x - self.home_icon.get_width() / 2, self.button_home.y - self.home_icon.get_height() / 2))

        self.button_load_image.draw(self.window)
        self.button_cam_image.draw(self.window)
        self.button_solve.draw(self.window)
        
        pygame.display.update()

    def handle_click(self, event):
        """This method helps handle leff mouse clicks"""

        # check if load from file button is clicked
        if self.button_load_image.clicked(event):
            # open the dialog box to select file
            path = askopenfilename(filetypes=[("image", ".jpeg"),
                                              ("image", ".png"),
                                              ("image", ".jpg"),
                                              ("image", ".jpe"),
                                              ("image", ".bmp")])
            # if file was selected
            if len(path) != 0:
                # load the file for image processing
                sip = SudokuImageProcessing(fname=path)
                # detect puzzle matrix
                mat = sip.get_matrix()
                # if matrix not detected
                if mat is None:
                    # show error
                    tkinter.messagebox.showerror(title="Error",
                                                 message="Unable to load file, try with different image.")
                    return
                # get the dimensions of the puzzle
                _, (box_rows, box_cols) = sip.get_dimensions()
                # re-initialize the object with the new data
                self.__init__(mat, box_rows, box_cols)

                # save the current puzzle loaded
                np.save('last_loaded.npy', self.matrix)
                # save the dimensions of the current puzzle
                np.save('last_loaded_dim.npy', np.array([self.BOX_ROWS, self.BOX_COLS]))
                # restart game
                self.play_game()

        # check if load from camera button is clicked
        if self.button_cam_image.clicked(event):
            # open the camera feed window
            with CameraWindow(capture_image=True) as feed:
                # while button is not pressed
                while True:
                    # frame is None until the button is clicked
                    frame = feed.draw_window()
                    if frame is not None:
                        break
            # intenstional delay to avoid accidental mouse clicks
            sleep(0.5)
            try:
                # load the image for image processing
                sip = SudokuImageProcessing(image=frame)
                # detect puzzle matrix
                mat = sip.get_matrix()
                # if matrix not detected
                if mat is None:
                    # raise exception
                    raise RuntimeError()
                # get the dimensions of the puzzle
                _, (box_rows, box_cols) = sip.get_dimensions()
                # re-initialize the object with the new data
                self.__init__(mat, box_rows, box_cols)
                # save the current puzzle loaded
                np.save('last_loaded.npy', self.matrix)
                # save the dimensions of the current puzzle
                np.save('last_loaded_dim.npy', np.array([self.BOX_ROWS, self.BOX_COLS]))
                # restart game
                self.play_game()

            except RuntimeError:
                # show error
                tkinter.messagebox.showerror(title="Error", message="Image not clear, please try again.")
                # reset screen size back to default game screen size
                self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                return

        # check if solve button is clicked
        if self.button_solve.clicked(event):
            # reset the matrix to inital state, i.e. remove all current entries
            self.matrix = self.init_matrix.copy()
            # while not solved
            while 0 in self.matrix:
                # find positions of empty cells
                rows, cols = np.where(self.matrix == 0)
                # choose a random coordinate of an empty cell
                coords = choice(list(zip(rows, cols)))
                # fill the cell with the solution value
                self.matrix[coords] = self.solution[coords]
                # delay to better visualize
                sleep(0.1)
                # draw the entries with green color
                self.draw_window(solved=True)

            # status variable
            hold_screen = True
            # while not action performed stay on screen
            while hold_screen:
                # iterate through events
                for event in pygame.event.get():
                    # if event is of type key press or button click
                    if event.type in (pygame.KEYDOWN, pygame.QUIT, pygame.MOUSEBUTTONDOWN):
                        # break outer while loop
                        hold_screen = False

        # check if home button is clicked
        if self.button_home.clicked(event):
            # go back to main menu
            return False

        # if LEFT mouse button is clicked
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # get the current coordinates of the mouse
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # convert to integer 
            mouse_x, mouse_y = int(mouse_x), int(mouse_y)
            # if mouse coordinates during the click are in the range of the game area
            if mouse_x in range(self.TOP_LEFT[X], self.TOP_LEFT[X] + self.NUM_COLUMNS * BLOCK_SIZE) and \
                    mouse_y in range(self.TOP_LEFT[Y], self.TOP_LEFT[Y] + self.NUM_ROWS * BLOCK_SIZE):
                # select the box in which the mouse is clicked
                self.selected_box = ((mouse_x - self.TOP_LEFT[X]) // BLOCK_SIZE,
                                     (mouse_y - self.TOP_LEFT[Y]) // BLOCK_SIZE)

        # default return is True
        return True

    def play_game(self):
        """This method handles the input part of the Game"""

        # display the game GUI
        self.draw_window()
        # iterate through events
        for event in pygame.event.get():

            # check for any click, return True if home is pressed
            if not self.handle_click(event):
                # return to the main menu
                return False

            # kill game, if window is closed or escape key is pressed
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                # save and exit
                self.graceful_exit()

            # if a key is pressed
            if event.type == pygame.KEYDOWN:
                # get coordinates of selected box
                box_i, box_j = self.selected_box
                # if up arrow key 
                if event.key == pygame.K_UP:
                    # shift the selected box up
                    box_j -= 1

                # if down arrow key 
                if event.key == pygame.K_DOWN:
                    # shift the selected box down
                    box_j += 1

                # if right arrow key 
                if event.key == pygame.K_RIGHT:
                    # shift the selected box right
                    box_i += 1

                # if left arrow key 
                if event.key == pygame.K_LEFT:
                    # shift the selected box left
                    box_i -= 1

                # update the selected box
                self.selected_box = (box_i % self.NUM_ROWS, box_j % self.NUM_COLUMNS)

        # check for keys pressed
        keys = pygame.key.get_pressed()
        # if escape key is pressed
        if keys[pygame.K_ESCAPE]:
            # save and exit
            self.graceful_exit()

        # get coordinates of selected box
        box_i, box_j = self.selected_box
        # iterate throught the number keys 
        for i in range(pygame.K_0, pygame.K_0 + self.NUM_ROWS + 1):
            # if key is pressed and the box does not contain a clue
            if keys[i] and (box_j, box_i) not in self.locked_pos:
                # fill the numeric value of the key pressed
                self.matrix[(box_j, box_i)] = i - pygame.K_0

        # iterate throught the numpad keys 
        for i in range(pygame.K_KP0, pygame.K_KP0 + self.NUM_ROWS + 1):
            # if key is pressed and the box does not contain a clue
            if keys[i] and (box_j, box_i) not in self.locked_pos:
                # fill the numeric value of the key pressed
                self.matrix[(box_j, box_i)] = i - pygame.K_KP0

        # if delete key is pressed and the box does not contain a clue
        if keys[pygame.K_DELETE] and (box_j, box_i) not in self.locked_pos:
            # remove the value
            self.matrix[(box_j, box_i)] = 0

        # if the current matrix is matches solution
        if np.array_equal(self.matrix, self.solution):
            # draw the keys as green
            self.draw_window(solved=True)

            # status variable
            hold_screen = True
            # while not action performed stay on screen
            while hold_screen:
                # iterate through events
                for event in pygame.event.get():
                    # if event is of type key press or button click
                    if event.type in (pygame.KEYDOWN, pygame.QUIT, pygame.MOUSEBUTTONDOWN):
                        # break outer while loop
                        hold_screen = False

            # fill backgroud with gray
            self.window.fill((100, 100, 100))
            # font for Heading
            font = pygame.font.SysFont('comicsans', 48)
            # label for heading
            label = font.render('Congratulations !!!', 1, (0, 200, 0))
            # display heading
            self.window.blit(label,
                             ((SCREEN_WIDTH - label.get_width()) / 2,
                              (SCREEN_HEIGHT - label.get_height()) / 2))
            # update the pygame display screen
            pygame.display.update()
            # save and exit
            self.graceful_exit()

        # True means game not over
        return True

    def load_AR(self):
        """This method handles the augmented reality solver"""

        # open the camera feed window
        with CameraWindow() as feed:
            # initialize frame count
            i = 0
            # initialize solution
            solution = None
            # infinite loop
            while True:
                # supress all kinds of exception
                with suppress(Exception):
                    # get frame from the camera
                    frame = feed.get_frame()
                    # every 10th frame
                    if i == 0:
                        # load the image for image processing
                        sip = SudokuImageProcessing(image=frame)
                        # detect puzzle matrix
                        detected_matrix = sip.get_matrix()
                        # if matrix detected
                        if detected_matrix is not None:
                            # get dimensions of the puzzle
                            game_size, (box_rows, box_cols) = sip.get_dimensions()
                            # minimum number of clues required to get a unqiue solution
                            min_clues_required = {4: 4, 6: 8, 8: 14, 9: 17}
                            # check for the number of clues given
                            if np.count_nonzero(detected_matrix) >= min_clues_required[game_size]:
                                # get list of solutions
                                solution_list = Sudoku(detected_matrix.copy(),
                                                       box_row=box_rows,
                                                       box_col=box_cols).get_solution()
                                # if atleast one solution is available
                                if len(solution_list) != 0:
                                    # store the solution
                                    solution = solution_list[0]
                    # for every frame, if a solution is found
                    if solution is not None:
                        # plot the solution on the frame
                        frame = sip.plot_on_image(frame, detected_matrix, solution, (box_rows, box_cols))
                # if home button is pressed
                if feed.draw_window(frame) is False:
                    # return to main menu
                    return False
                # increment frame count
                i = (i + 1) % 10

    def main_menu(self):
        """Shows the menu for the program"""

        # fill the background
        self.window.fill((255, 255, 255))
        # font for the heading
        font = pygame.font.SysFont('comicsans', 60)
        # label for the heading
        label = font.render('VisioNxN Sudoku', 1, (0, 0, 0))
        # display the heading
        self.window.blit(label,
                         ((SCREEN_WIDTH - label.get_width()) / 2,
                          100 - label.get_height() / 2))

        # display the play game button
        self.button_play_game.draw(self.window)
        # display the augmented reality button
        self.button_AR.draw(self.window)

        # iterate through events
        for event in pygame.event.get():
            # kill game, if window is closed or escape key is pressed
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                # save and exit
                self.graceful_exit()

            # if play game button is pressed
            if self.button_play_game.clicked(event):
                return 1

            # if augmented reality button is pressed
            if self.button_AR.clicked(event):
                return 2

        # update the pygame display screen
        pygame.display.update()
        # return that no option is selected
        return 0

    def graceful_exit(self):
        """Helper method, it saves the last loaded puzzle and its dimensions before quitting"""
        # save the current puzzle loaded
        np.save('last_loaded.npy', self.init_matrix)
        # save the dimensions of the current puzzle
        np.save('last_loaded_dim.npy', np.array([self.BOX_ROWS, self.BOX_COLS]))
        # exit pygame runtime
        pygame.quit()
        # exit program
        quit()
