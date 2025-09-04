"""
Camera system for following the player and handling manual zoom
"""

import pygame
from pygame.math import Vector2
from typing import Tuple
from utils.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    WORLD_WIDTH,
    WORLD_HEIGHT,
    CAMERA_MIN_ZOOM,
)
from utils.math_utils import clamp_value


class Camera:
    """Camera system that follows the player and handles manual zoom"""

    def __init__(self):
        # Camera position in world coordinates
        self.x = 0
        self.y = 0

        # Manual zoom factor (controlled by mouse wheel)
        self.zoom_factor = 1.0

        # Smoothing for camera movement only
        self.position_smoothness = 0.1

        # Effective screen dimensions (with zoom)
        self.effective_width = SCREEN_WIDTH
        self.effective_height = SCREEN_HEIGHT

        # Store the world position that should stay at screen center during zoom
        self.zoom_center_world_pos = Vector2(0, 0)

    def handle_mouse_wheel(self, wheel_y: int):
        """Handle mouse wheel zoom input"""
        # Store current zoom and effective dimensions before change
        old_zoom = self.zoom_factor
        old_effective_width = self.effective_width
        old_effective_height = self.effective_height

        # Zoom in/out based on wheel direction
        zoom_change = 0.1 if wheel_y > 0 else -0.1
        new_zoom = self.zoom_factor + zoom_change

        # Clamp zoom to reasonable bounds
        self.zoom_factor = clamp_value(new_zoom, CAMERA_MIN_ZOOM, 3.0)

        # Update effective screen dimensions
        self.effective_width = int(SCREEN_WIDTH / self.zoom_factor)
        self.effective_height = int(SCREEN_HEIGHT / self.zoom_factor)

        # Calculate how much the viewport size changed
        width_change = self.effective_width - old_effective_width
        height_change = self.effective_height - old_effective_height

        # Adjust camera position to compensate for the size change
        # This keeps the same world point centered on screen
        self.x += width_change / 2
        self.y += height_change / 2

        # Ensure we're still within world boundaries
        self._clamp_to_world_boundaries()

    def update(self, target_position: Vector2, dt: float):
        """Update camera position smoothly (zoom is handled manually)"""
        # Store the world position that should stay at screen center
        self.zoom_center_world_pos = target_position

        # Calculate target camera position (center on target)
        target_camera_x = target_position.x - (self.effective_width / 2)
        target_camera_y = target_position.y - (self.effective_height / 2)

        # Smoothly move camera towards target
        self.x += (target_camera_x - self.x) * self.position_smoothness
        self.y += (target_camera_y - self.y) * self.position_smoothness

        # Clamp to world boundaries
        self._clamp_to_world_boundaries()

        # Ensure camera position is stable (prevent micro-movements)
        if abs(self.x - target_camera_x) < 1.0:
            self.x = target_camera_x
        if abs(self.y - target_camera_y) < 1.0:
            self.y = target_camera_y

    def _clamp_to_world_boundaries(self):
        """Clamp camera position to world boundaries"""
        # Left boundary
        if self.x < 0:
            self.x = 0

        # Right boundary
        if self.x + self.effective_width > WORLD_WIDTH:
            self.x = WORLD_WIDTH - self.effective_width

        # Top boundary
        if self.y < 0:
            self.y = 0

        # Bottom boundary
        if self.y + self.effective_height > WORLD_HEIGHT:
            self.y = WORLD_HEIGHT - self.effective_height

    def world_to_screen(self, world_pos: Vector2) -> Vector2:
        """Convert world coordinates to screen coordinates"""
        screen_x = (world_pos.x - self.x) * self.zoom_factor
        screen_y = (world_pos.y - self.y) * self.zoom_factor
        return Vector2(screen_x, screen_y)

    def screen_to_world(self, screen_pos: Vector2) -> Vector2:
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_pos.x / self.zoom_factor) + self.x
        world_y = (screen_pos.y / self.zoom_factor) + self.y
        return Vector2(world_x, world_y)

    def get_viewport_rect(self) -> pygame.Rect:
        """Get the current viewport rectangle in world coordinates"""
        return pygame.Rect(self.x, self.y, self.effective_width, self.effective_height)

    def is_visible(self, world_pos: Vector2, radius: float = 0) -> bool:
        """Check if a world position is visible in the current viewport"""
        viewport = self.get_viewport_rect()

        if radius == 0:
            return viewport.collidepoint(world_pos.x, world_pos.y)
        else:
            # Check if circle intersects with viewport
            return viewport.colliderect(
                pygame.Rect(
                    world_pos.x - radius, world_pos.y - radius, radius * 2, radius * 2
                )
            )

    def get_zoom_info(self) -> dict:
        """Get information about current zoom state"""
        return {
            "zoom_factor": self.zoom_factor,
            "effective_width": self.effective_width,
            "effective_height": self.effective_height,
            "world_visible_width": self.effective_width,
            "world_visible_height": self.effective_height,
        }

    def reset(self, target_position: Vector2):
        """Reset camera to center on target"""
        self.x = target_position.x - (SCREEN_WIDTH / 2)
        self.y = target_position.y - (SCREEN_HEIGHT / 2)
        self.zoom_factor = 1.0
        self.effective_width = SCREEN_WIDTH
        self.effective_height = SCREEN_HEIGHT
        self.zoom_center_world_pos = target_position
        self._clamp_to_world_boundaries()

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Camera(pos=({self.x:.1f}, {self.y:.1f}), zoom={self.zoom_factor:.3f})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"Camera(pos=({self.x:.1f}, {self.y:.1f}), zoom={self.zoom_factor:.3f}, effective=({self.effective_width}, {self.effective_height}))"
