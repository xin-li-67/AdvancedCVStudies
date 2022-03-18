import os
import cv2
import pygame

import numpy as np

from button import Button

os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()

class CameraWindow:
    def __init__(self, cam_source: int = 0, capture_image: bool = False):
        """default initialization"""
        self.cam_source = cam_source
        self.capture_image = capture_image

    def __enter__(self):
        """context manager `__enter__` method invoked on entry to `with` statement."""
        self.camera = cv2.VideoCapture(self.cam_source)
        assert self.camera is not None, "Camera not available, try using a different cam_source value"
        # read the first frame to get the shape of the image
        _, frame = self.camera.read()
        rows, cols, _ = frame.shape
        # if window meant to capture image
        if self.capture_image:
            # add additional height to the pygame screen for the click button
            self.screen = pygame.display.set_mode((cols, rows + 100))
            self.camera_icon = pygame.image.load('../images/camera_icon.png')
            self.button_click = Button(cols // 2, rows + 50, 250, 60, (200, 200, 200), '  ')
        else:
            # create a screen with the size same as the image
            self.screen = pygame.display.set_mode((cols, rows))
            self.home_icon = pygame.image.load('../images/home_icon.png')
            self.button_home = Button(60, 60, 70, 70, (200, 200, 200), '  ')
        
        return self

    def get_frame(self):
        """get the current frame from the screen"""
        _, frame = self.camera.read()
        
        return frame

    def draw_window(self, frame: np.ndarray = None):
        """draw a window the current camera feed"""
        self.screen.fill((255, 255, 255))
        
        if frame is None:
            _, frame = self.camera.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.surfarray.make_surface(np.flip(np.rot90(frame.copy()), axis=0))
        self.screen.blit(pygame_frame, (0, 0))
        
        if self.capture_image:
            self.button_click.draw(self.screen)
            self.screen.blit(self.camera_icon,
                             (self.button_click.x - self.camera_icon.get_width() / 2,
                              self.button_click.y - self.camera_icon.get_height() / 2))
        else:
            self.button_home.draw(self.screen)
            self.screen.blit(self.home_icon,
                             (self.button_home.x - self.home_icon.get_width() / 2,
                              self.button_home.y - self.home_icon.get_height() / 2))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # if click button is pressed 
            elif self.capture_image and self.button_click.clicked(event):
                # return the current image frame
                return frame

            # if home button is pressed
            elif not self.capture_image and self.button_home.clicked(event):
                # return to main menu
                return False
        
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """context manager `__exit__` method invoked during exit from `with` statement."""
        self.camera.release()