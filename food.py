import pygame


class Food(pygame.sprite.Sprite):

    def __init__(self, surface, color, center, radius):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = center

        self.surface = surface
        self.color = color
        self.center = center
        self.radius = radius

    # def draw(self, surface):

    # def update(self):
    # if self.rect.collideobjects():
    # self.kill()
