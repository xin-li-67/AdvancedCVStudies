import pygame

from typing import Tuple

pygame.font.init()

class Button:
    def __init__(self, x: int, y: int, button_width: int, button_height: int, color: Tuple, text: str):
        self.x = x
        self.y = y
        self.button_width = button_width
        self.button_height = button_height
        self.color = color
        self.hover_color = tuple(3 * (c // 4) for c in color)
        self.radius = 0.5
        self.text = text

    def draw(self, surface: pygame.Surface):
        """used to draw the button on a surface"""
        rect = pygame.Rect(self.x - self.button_width // 2,
                           self.y - self.button_height // 2,
                           self.button_width,
                           self.button_height)

        # check if current mouse position is over the button area
        if self.under_mouse():
            # set fill color
            color = pygame.Color(*self.hover_color)
        else:
            # otherwise darken the button
            color = pygame.Color(*self.color)
        # specifies opacity of the color
        alpha = color.a
        # alpha componenent of pygame Color object
        color.a = 0

        pos = rect.topleft
        rect.topleft = 0, 0

        rectangle = pygame.Surface(rect.size, pygame.SRCALPHA)
        circle = pygame.Surface([min(rect.size) * 3] * 2, pygame.SRCALPHA)
        pygame.draw.ellipse(circle, (0, 0, 0), circle.get_rect(), 0)
        circle = pygame.transform.smoothscale(circle, [int(min(rect.size) * self.radius)] * 2)

        radius = rectangle.blit(circle, (0, 0))
        radius.bottomright = rect.bottomright
        rectangle.blit(circle, radius)
        radius.topright = rect.topright
        rectangle.blit(circle, radius)
        radius.bottomleft = rect.bottomleft
        rectangle.blit(circle, radius)

        rectangle.fill((0, 0, 0), rect.inflate(-radius.w, 0))
        rectangle.fill((0, 0, 0), rect.inflate(0, -radius.h))
        rectangle.fill(color, special_flags=pygame.BLEND_RGBA_MAX)
        rectangle.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MIN)

        font = pygame.font.SysFont('comicsans', self.button_height // 2)
        label = font.render(self.text, 1, (0, 0, 0))

        surface.blit(rectangle, pos)
        surface.blit(label, (self.x - label.get_width() // 2,
                             self.y - label.get_height() // 2))

    def clicked(self, event):
        """used to check if the button is clicked"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.under_mouse()

    def under_mouse(self):
        """find if the current mouse coordinates"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        # if mouse coordinates during the click are in the range of the button coordinate
        if mouse_x in range(self.x - self.button_width // 2, self.x + self.button_width // 2) and \
                mouse_y in range(self.y - self.button_height // 2, self.y + self.button_height // 2):
            
            return True
        return False