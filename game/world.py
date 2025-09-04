"""
World management system for the Agar.io game
"""

import pygame
from pygame.math import Vector2
from typing import List
from utils.constants import WORLD_WIDTH, WORLD_HEIGHT, BG_COLOR, BG_GRADIENT_COLOR


class World:
    """Manages the game world and gradient background"""

    def __init__(self):
        # World dimensions
        self.width = WORLD_WIDTH
        self.height = WORLD_HEIGHT

        # Create world surface
        self.surface = pygame.Surface((self.width, self.height))

        # Draw gradient background
        self._draw_gradient_background()

        # World boundaries
        self.boundaries = pygame.Rect(0, 0, self.width, self.height)

    def _draw_gradient_background(self):
        """Draw a smooth gradient background from dark to light"""
        # Start with the darker base color
        self.surface.fill(BG_COLOR)

        # Create gradient effect by drawing multiple horizontal lines with varying alpha
        for y in range(self.height):
            # Calculate alpha based on y position (darker at top, lighter at bottom)
            alpha_ratio = y / self.height
            alpha = int(255 * alpha_ratio)

            # Create a line with the gradient color and calculated alpha
            line_color = (*BG_GRADIENT_COLOR, alpha)

            # Draw horizontal line with alpha blending
            line_surface = pygame.Surface((self.width, 1))
            line_surface.fill(BG_GRADIENT_COLOR)
            line_surface.set_alpha(alpha)
            self.surface.blit(line_surface, (0, y))

    def draw(
        self,
        screen: pygame.Surface,
        camera_x: float,
        camera_y: float,
        zoom_factor: float = 1.0,
    ):
        """Draw the world background with camera offset and zoom"""
        # Calculate visible area
        screen_width = screen.get_width()
        screen_height = screen.get_height()

        # Calculate effective screen dimensions
        effective_width = int(screen_width / zoom_factor)
        effective_height = int(screen_height / zoom_factor)

        # Calculate source rectangle (what's visible in the world)
        source_x = max(0, int(camera_x))
        source_y = max(0, int(camera_y))
        source_width = min(effective_width, self.width - source_x)
        source_height = min(effective_height, self.height - source_y)

        # Ensure we don't go out of bounds
        if source_x + source_width > self.width:
            source_width = self.width - source_x
        if source_y + source_height > self.height:
            source_height = self.height - source_y

        # Ensure minimum dimensions
        if source_width <= 0 or source_height <= 0:
            return

        # Create source rectangle
        source_rect = pygame.Rect(source_x, source_y, source_width, source_height)

        # Create destination rectangle (where to draw on screen)
        dest_rect = pygame.Rect(0, 0, screen_width, screen_height)

        try:
            # Scale the source to fit the screen
            scaled_surface = pygame.transform.scale(
                self.surface.subsurface(source_rect), (screen_width, screen_height)
            )

            # Draw to screen
            screen.blit(scaled_surface, dest_rect)
        except (ValueError, pygame.error):
            # Fallback if subsurface fails - draw solid background
            screen.fill(BG_COLOR)

    def is_position_valid(self, position: Vector2, radius: float = 0) -> bool:
        """Check if a position is within world boundaries"""
        if position.x - radius < 0 or position.x + radius > self.width:
            return False
        if position.y - radius < 0 or position.y + radius > self.height:
            return False
        return True

    def clamp_position(self, position: Vector2, radius: float = 0) -> Vector2:
        """Clamp a position to world boundaries"""
        clamped_x = max(radius, min(position.x, self.width - radius))
        clamped_y = max(radius, min(position.y, self.height - radius))
        return Vector2(clamped_x, clamped_y)

    def get_random_position(self, radius: float = 0) -> Vector2:
        """Get a random position within world boundaries"""
        import random

        x = random.randint(radius, self.width - radius)
        y = random.randint(radius, self.height - radius)
        return Vector2(x, y)

    def get_center_position(self) -> Vector2:
        """Get the center position of the world"""
        return Vector2(self.width / 2, self.height / 2)

    def get_boundaries(self) -> pygame.Rect:
        """Get world boundaries"""
        return self.boundaries

    def get_size(self) -> tuple:
        """Get world dimensions"""
        return (self.width, self.height)

    def get_area(self) -> float:
        """Get world area"""
        return self.width * self.height

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"World({self.width}x{self.height})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"World({self.width}x{self.height}, area={self.get_area()})"
