"""
Camera system for following the player and handling zoom
"""

import pygame
from pygame.math import Vector2
from typing import Tuple
from utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT,
    CAMERA_MIN_ZOOM
)
from utils.math_utils import (
    calculate_zoom_factor, calculate_effective_screen_dimensions,
    clamp_value
)

class Camera:
    """Camera system that follows the player and handles zoom"""
    
    def __init__(self):
        # Camera position in world coordinates
        self.x = 0
        self.y = 0
        
        # Zoom factor
        self.zoom_factor = 1.0
        self.target_zoom_factor = 1.0
        
        # Smoothing
        self.zoom_smoothness = 0.1
        self.position_smoothness = 0.1
        
        # Effective screen dimensions (with zoom)
        self.effective_width = SCREEN_WIDTH
        self.effective_height = SCREEN_HEIGHT
        
    def update(self, target_position: Vector2, target_zoom: float, dt: float):
        """Update camera position and zoom smoothly"""
        # Update zoom factor
        self.target_zoom_factor = target_zoom
        self.zoom_factor += (self.target_zoom_factor - self.zoom_factor) * self.zoom_smoothness
        self.zoom_factor = max(CAMERA_MIN_ZOOM, self.zoom_factor)
        
        # Calculate effective screen dimensions
        self.effective_width, self.effective_height = calculate_effective_screen_dimensions(
            SCREEN_WIDTH, SCREEN_HEIGHT, self.zoom_factor
        )
        
        # Calculate target camera position (center on target)
        target_camera_x = target_position.x - (self.effective_width / 2)
        target_camera_y = target_position.y - (self.effective_height / 2)
        
        # Smoothly move camera towards target
        self.x += (target_camera_x - self.x) * self.position_smoothness
        self.y += (target_camera_y - self.y) * self.position_smoothness
        
        # Clamp to world boundaries
        self._clamp_to_world_boundaries()
        
        # Ensure camera position is stable (prevent micro-movements that cause grid glitches)
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
        return pygame.Rect(
            self.x, self.y,
            self.effective_width, self.effective_height
        )
    
    def is_visible(self, world_pos: Vector2, radius: float = 0) -> bool:
        """Check if a world position is visible in the current viewport"""
        viewport = self.get_viewport_rect()
        
        if radius == 0:
            return viewport.collidepoint(world_pos.x, world_pos.y)
        else:
            # Check if circle intersects with viewport
            return viewport.colliderect(pygame.Rect(
                world_pos.x - radius, world_pos.y - radius,
                radius * 2, radius * 2
            ))
    
    def get_zoom_info(self) -> dict:
        """Get information about current zoom state"""
        return {
            'zoom_factor': self.zoom_factor,
            'effective_width': self.effective_width,
            'effective_height': self.effective_height,
            'world_visible_width': self.effective_width,
            'world_visible_height': self.effective_height
        }
    
    def reset(self, target_position: Vector2):
        """Reset camera to center on target"""
        self.x = target_position.x - (SCREEN_WIDTH / 2)
        self.y = target_position.y - (SCREEN_HEIGHT / 2)
        self.zoom_factor = 1.0
        self.target_zoom_factor = 1.0
        self._clamp_to_world_boundaries()
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Camera(pos=({self.x:.1f}, {self.y:.1f}), zoom={self.zoom_factor:.3f})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"Camera(pos=({self.x:.1f}, {self.y:.1f}), zoom={self.zoom_factor:.3f}, effective=({self.effective_width}, {self.effective_height}))"
