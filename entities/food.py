"""
Enhanced Food class for the Agar.io game
"""

import pygame
import random
from typing import Optional
from utils.constants import FOOD_SIZES, BG_COLOR, WORLD_WIDTH, WORLD_HEIGHT
from utils.math_utils import get_random_point_in_rect
import math


class Food(pygame.sprite.Sprite):
    """Enhanced food class with improved collision detection and spawning"""

    def __init__(
        self,
        world_surface=None,
        color: Optional[tuple] = None,
        center: Optional[tuple] = None,
        size: Optional[float] = None,
    ):
        super().__init__()

        # Generate random properties if not provided
        if size is None:
            size = self._generate_food_size()
        if color is None:
            color = self._generate_random_color()
        if center is None:
            center = self._generate_random_center(world_surface)

        # Store properties
        self.size = size
        self.score_value = size
        self.color = color
        self.center = center

        # Create visual representation
        self.image = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (size, size), size)

        # Set up collision rectangle
        self.rect = self.image.get_rect()
        self.rect.center = center

        # Store world reference
        self.world_surface = world_surface

        # Animation properties
        self.pulse_angle = 0
        self.pulse_speed = 0.1

    def _generate_food_size(self) -> float:
        """Generate food size based on probability distribution"""
        rand = random.random()
        cumulative_prob = 0

        for size, prob in FOOD_SIZES.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return size

        return 10.0  # Fallback to smallest size

    def _generate_random_color(self) -> pygame.Color:
        """Generate a random color that's different from background"""
        while True:
            r, g, b = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            color = (r, g, b)

            if color != BG_COLOR:
                return pygame.Color(r, g, b)

    def _generate_random_center(self, world_surface) -> tuple:
        """Generate random center position within world bounds"""
        if world_surface is None:
            # Use default world bounds if no surface provided
            x = random.randint(100, WORLD_WIDTH - 100)
            y = random.randint(100, WORLD_HEIGHT - 100)
            return (x, y)
        else:
            world_rect = world_surface.get_rect()
            return get_random_point_in_rect(world_rect)

    def update(self, dt: float):
        """Update food animation and effects"""
        # Simple pulse animation
        self.pulse_angle += self.pulse_speed * dt
        if self.pulse_angle > 2 * 3.14159:  # 2π
            self.pulse_angle = 0

    def draw(
        self,
        surface: pygame.Surface,
        camera_x: float,
        camera_y: float,
        zoom_factor: float = 1.0,
    ):
        """Draw food with camera offset and zoom scaling"""
        # Calculate screen position
        screen_x = (self.rect.centerx - camera_x) * zoom_factor
        screen_y = (self.rect.centery - camera_y) * zoom_factor

        # Scale the food size by zoom factor
        scaled_size = int(self.size * zoom_factor)

        # Create a scaled surface for the food
        scaled_surface = pygame.transform.scale(
            self.image, (scaled_size * 2, scaled_size * 2)
        )

        # Apply pulse effect
        pulse_factor = 1.0 + 0.1 * math.sin(self.pulse_angle)
        final_size = int(scaled_size * pulse_factor)

        # Ensure final size is reasonable
        final_size = max(2, min(final_size, scaled_size * 2))

        # Create final surface
        final_surface = pygame.transform.scale(
            self.image, (final_size * 2, final_size * 2)
        )

        # Draw to screen
        surface.blit(final_surface, (screen_x - final_size, screen_y - final_size))

    def respawn(self, world_surface=None):
        """Respawn food at a new random location"""
        new_center = self._generate_random_center(world_surface)
        self.rect.center = new_center
        self.center = new_center

        # Optionally regenerate size and color
        if random.random() < 0.3:  # 30% chance to change properties
            self.size = self._generate_food_size()
            self.score_value = self.size
            self.color = self._generate_random_color()

            # Recreate image
            self.image = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                self.image, self.color, (self.size, self.size), self.size
            )

    def get_collision_rect(self) -> pygame.Rect:
        """Get collision rectangle for precise collision detection"""
        return self.rect

    def is_consumable_by(self, player_size: float) -> bool:
        """Check if this food can be consumed by a player of given size"""
        return player_size > self.size

    def get_consumption_requirement(self) -> float:
        """Get the minimum size required to consume this food"""
        return self.size + 1  # Must be at least 1 size larger

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Food(size={self.size}, score={self.score_value}, pos={self.center})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"Food(size={self.size}, score={self.score_value}, pos={self.center}, color={self.color})"
