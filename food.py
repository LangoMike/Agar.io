import pygame
import random


class Food(pygame.sprite.Sprite):

    def __init__(self, surface, color, center, size):
        pygame.sprite.Sprite.__init__(self)
        # Size determines both the visual size and the score value
        self.size = size
        self.score_value = size
        
        # Create a surface that can handle transparency
        self.image = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        # Draw a circle with size proportional to score value
        pygame.draw.circle(self.image, color, (size, size), size)
        self.rect = self.image.get_rect()
        self.rect.center = center

        self.surface = surface
        self.color = color
        self.center = center

    def draw(self, surface, camera_x=0, camera_y=0, zoom_factor=1.0):
        """Draw food with camera offset and zoom scaling"""
        screen_x = (self.rect.centerx - camera_x) * zoom_factor
        screen_y = (self.rect.centery - camera_y) * zoom_factor
        # Scale the food size by zoom factor
        scaled_size = int(self.size * zoom_factor)
        # Create a scaled surface for the food
        scaled_surface = pygame.transform.scale(self.image, (scaled_size * 2, scaled_size * 2))
        surface.blit(scaled_surface, (screen_x - scaled_size, screen_y - scaled_size))

    def update(self):
        pass
