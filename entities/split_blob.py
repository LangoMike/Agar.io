"""
SplitBlob class for individual blob behavior in split mode
"""

import pygame
import math
from pygame.math import Vector2
from typing import Optional, List
from utils.constants import PLAYER_COLOR, PLAYER_BASE_SPEED, PLAYER_SPEED_POWER
from utils.math_utils import (
    distance_between_points, normalize_vector, clamp_value,
    point_in_circle, circles_intersect
)

class SplitBlob:
    """Individual blob that can move, eat food, and interact with other entities"""
    
    def __init__(self, position: Vector2, size: float, color: tuple = PLAYER_COLOR):
        self.position = Vector2(position)
        self.size = size
        self.color = color
        self.original_size = size  # Track original size for rejoining
        
        # Movement properties
        self.velocity = Vector2(0, 0)
        self.target_position = Vector2(position)
        self.speed = self._calculate_speed()
        
        # Collision properties
        self.collision_radius = size
        self.collision_rect = pygame.Rect(
            position.x - size, position.y - size,
            size * 2, size * 2
        )
        
        # State properties
        self.is_active = True
        self.food_eaten = []  # Track food consumed by this blob
        
        # Visual properties
        self.pulse_angle = 0
        self.pulse_speed = 0.15
        
        # Split movement properties
        self.split_offset = Vector2(0, 0)  # Offset from main blob
        self.split_angle = 0  # Angle around main blob
        
    def _calculate_speed(self) -> float:
        """Calculate movement speed based on size"""
        return PLAYER_BASE_SPEED / max(1, self.size ** PLAYER_SPEED_POWER)
    
    def update(self, dt: float, target_position: Vector2, other_blobs: List = None):
        """Update blob position and state"""
        if not self.is_active:
            return
            
        # Update target position
        self.target_position = Vector2(target_position)
        
        # Calculate base movement towards target
        direction = self.target_position - self.position
        if direction.length() > 0:
            direction = normalize_vector(direction)
            
            # Calculate base movement
            base_movement = direction * self.speed * dt
            
            # Apply collision avoidance with other blobs
            avoidance_movement = self._calculate_avoidance_movement(other_blobs or [])
            
            # Combine movements
            final_movement = base_movement + avoidance_movement
            
            # Move the blob
            self.position += final_movement
            
            # Update collision properties
            self._update_collision_properties()
            
            # Update pulse animation
            self.pulse_angle += self.pulse_speed * dt
            if self.pulse_angle > 2 * math.pi:  # 2π
                self.pulse_angle = 0
    
    def _calculate_avoidance_movement(self, other_blobs: List) -> Vector2:
        """Calculate movement to avoid overlapping with other blobs"""
        avoidance = Vector2(0, 0)
        
        for other_blob in other_blobs:
            if other_blob == self or not other_blob.is_active:
                continue
                
            # Calculate distance between blob centers
            distance = distance_between_points(self.position, other_blob.position)
            min_distance = self.size + other_blob.size
            
            # If blobs are too close, push them apart
            if distance < min_distance and distance > 0:
                # Calculate push direction (away from other blob)
                push_direction = (self.position - other_blob.position).normalize()
                
                # Calculate push strength (stronger when closer)
                push_strength = (min_distance - distance) * 0.5
                
                # Add to avoidance movement
                avoidance += push_direction * push_strength
        
        # Limit avoidance movement to prevent excessive pushing
        if avoidance.length() > self.speed * 0.1:  # Max 10% of speed
            avoidance = avoidance.normalize() * self.speed * 0.1
            
        return avoidance
    
    def set_split_position(self, main_position: Vector2, angle: float, distance: float):
        """Set the blob's position relative to the main blob"""
        self.split_angle = angle
        self.split_offset = Vector2(
            math.cos(angle) * distance,
            math.sin(angle) * distance
        )
        self.position = main_position + self.split_offset
    
    def _update_collision_properties(self):
        """Update collision rectangle and radius based on current position"""
        self.collision_radius = self.size
        self.collision_rect.center = self.position
    
    def can_eat_food(self, food) -> bool:
        """Check if this blob can eat the given food"""
        if not self.is_active:
            return False
        return self.size > food.size
    
    def can_eat_enemy(self, enemy) -> bool:
        """Check if this blob can eat the given enemy"""
        if not self.is_active:
            return False
        return self.size > enemy.size * 1.01  # Must be 1% larger
    
    def can_eat_player_blob(self, other_blob) -> bool:
        """Check if this blob can eat another player blob"""
        if not self.is_active or not other_blob.is_active:
            return False
        return self.size > other_blob.size * 1.01  # Must be 1% larger
    
    def eat_food(self, food):
        """Consume food and grow"""
        if self.can_eat_food(food):
            self.size += food.score_value
            self.food_eaten.append(food)
            self.speed = self._calculate_speed()  # Recalculate speed
            return True
        return False
    
    def eat_enemy(self, enemy):
        """Consume enemy and grow"""
        if self.can_eat_enemy(enemy):
            self.size += enemy.size * 0.8  # Get 80% of enemy mass
            self.speed = self._calculate_speed()
            return True
        return False
    
    def eat_player_blob(self, other_blob):
        """Consume another player blob and grow"""
        if self.can_eat_player_blob(other_blob):
            self.size += other_blob.size * 0.9  # Get 90% of blob mass
            self.speed = self._calculate_speed()
            return True
        return False
    
    def check_collision_with_food(self, food) -> bool:
        """Check collision with food (center touch required)"""
        if not self.is_active:
            return False
        
        # Check if blob center touches food center
        distance = distance_between_points(self.position, Vector2(food.rect.center))
        return distance <= self.size
    
    def check_collision_with_enemy(self, enemy) -> bool:
        """Check collision with enemy blob"""
        if not self.is_active:
            return False
        
        return circles_intersect(
            self.position, self.size,
            enemy.position, enemy.size
        )
    
    def check_collision_with_player_blob(self, other_blob) -> bool:
        """Check collision with another player blob"""
        if not self.is_active or not other_blob.is_active:
            return False
        
        return circles_intersect(
            self.position, self.size,
            other_blob.position, other_blob.size
        )
    
    def draw(self, surface: pygame.Surface, camera_x: float, camera_y: float, 
             zoom_factor: float = 1.0):
        """Draw blob with camera offset and zoom scaling"""
        if not self.is_active:
            return
            
        # Calculate screen position
        screen_x = (self.position.x - camera_x) * zoom_factor
        screen_y = (self.position.y - camera_y) * zoom_factor
        
        # Calculate scaled size
        scaled_size = int(self.size * zoom_factor)
        
        # Apply pulse effect
        pulse_factor = 1.0 + 0.05 * math.sin(self.pulse_angle)
        final_size = int(scaled_size * pulse_factor)
        
        # Ensure reasonable size
        final_size = max(2, min(final_size, scaled_size * 2))
        
        # Draw blob
        pygame.draw.circle(
            surface,
            self.color,
            (int(screen_x), int(screen_y)),
            final_size
        )
        
        # Draw outline for better visibility
        pygame.draw.circle(
            surface,
            (255, 255, 255),  # White outline
            (int(screen_x), int(screen_y)),
            final_size,
            2  # 2 pixel outline
        )
    
    def get_total_mass_gained(self) -> float:
        """Get total mass gained from eating food"""
        return sum(food.score_value for food in self.food_eaten)
    
    def reset_food_tracking(self):
        """Reset food eaten tracking (called after rejoining)"""
        self.food_eaten.clear()
    
    def deactivate(self):
        """Deactivate this blob (for rejoining)"""
        self.is_active = False
    
    def activate(self):
        """Activate this blob"""
        self.is_active = True
    
    def get_mass_for_rejoin(self) -> float:
        """Get mass that should be added to main player when rejoining"""
        return self.size - self.original_size
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return f"SplitBlob(size={self.size:.1f}, pos={self.position}, active={self.is_active})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"SplitBlob(size={self.size:.1f}, pos={self.position}, active={self.is_active}, food_eaten={len(self.food_eaten)})"
