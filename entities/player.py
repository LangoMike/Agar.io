"""
Player class that manages the main player and split blobs
"""

import pygame
import time
import math
from pygame.math import Vector2
from typing import List, Optional, Tuple
from utils.constants import (
    PLAYER_START_SIZE, PLAYER_MIN_SIZE, PLAYER_MAX_SPLITS,
    PLAYER_SPLIT_REJOIN_TIME, PLAYER_COLOR, WORLD_WIDTH, WORLD_HEIGHT
)
from utils.math_utils import (
    distance_between_points, normalize_vector, clamp_value,
    calculate_zoom_factor
)
from .split_blob import SplitBlob

class Player:
    """Main player class that manages the player blob and split functionality"""
    
    def __init__(self, start_position: Vector2, start_size: float = PLAYER_START_SIZE):
        # Main player properties
        self.position = Vector2(start_position)
        self.size = start_size
        self.color = PLAYER_COLOR
        self.original_size = start_size
        
        # Split management
        self.split_blobs: List[SplitBlob] = []
        self.split_count = 0
        self.last_split_time = 0
        self.is_split = False
        
        # Movement properties
        self.velocity = Vector2(0, 0)
        self.target_position = Vector2(start_position)
        self.speed = self._calculate_speed()
        
        # Collision properties
        self.collision_radius = start_size
        self.collision_rect = pygame.Rect(
            start_position.x - start_size, start_position.y - start_size,
            start_size * 2, start_size * 2
        )
        
        # State properties
        self.is_active = True
        self.total_mass_gained = 0
        
        # Visual properties
        self.pulse_angle = 0
        self.pulse_speed = 0.1
        
    def _calculate_speed(self) -> float:
        """Calculate movement speed based on size"""
        from utils.constants import PLAYER_BASE_SPEED, PLAYER_SPEED_POWER
        return PLAYER_BASE_SPEED / max(1, self.size ** PLAYER_SPEED_POWER)
    
    def update(self, dt: float, target_position: Vector2):
        """Update player position and state"""
        if not self.is_active:
            return
            
        # Update target position
        self.target_position = Vector2(target_position)
        
        if self.is_split:
            # Update split blobs
            self._update_split_blobs(dt, target_position)
            
            # Check if it's time to rejoin
            if time.time() - self.last_split_time >= PLAYER_SPLIT_REJOIN_TIME:
                self.rejoin_blobs()
                self.is_split = False
                self.split_count = 0
                
        else:
            # Update main player
            self._update_main_player(dt)
            
            # Update collision properties
            self._update_collision_properties()
            
            # Update pulse animation
            self.pulse_angle += self.pulse_speed * dt
            if self.pulse_angle > 2 * math.pi:
                self.pulse_angle = 0
    
    def _update_main_player(self, dt: float):
        """Update main player movement"""
        # Calculate direction to target
        direction = self.target_position - self.position
        if direction.length() > 0:
            direction = normalize_vector(direction)
            
            # Move towards target
            movement = direction * self.speed * dt
            self.position += movement
            
            # Clamp to world boundaries
            self.position.x = clamp_value(self.position.x, self.size, WORLD_WIDTH - self.size)
            self.position.y = clamp_value(self.position.y, self.size, WORLD_HEIGHT - self.size)
    
    def _update_split_blobs(self, dt: float, target_position: Vector2):
        """Update all split blobs"""
        for blob in self.split_blobs:
            if blob.is_active:
                # Pass other blobs for collision avoidance
                other_blobs = [b for b in self.split_blobs if b != blob and b.is_active]
                blob.update(dt, target_position, other_blobs)
                
                # Clamp to world boundaries
                blob.position.x = clamp_value(blob.position.x, blob.size, WORLD_WIDTH - blob.size)
                blob.position.y = clamp_value(blob.position.y, blob.size, WORLD_HEIGHT - blob.size)
    
    def _update_collision_properties(self):
        """Update collision rectangle and radius based on current position"""
        self.collision_radius = self.size
        self.collision_rect.center = self.position
    
    def can_split(self) -> bool:
        """Check if player can split"""
        # Check split count limit
        if self.split_count >= PLAYER_MAX_SPLITS:
            return False
        
        # If not split yet, check main player size
        if not self.is_split:
            if self.size < PLAYER_MIN_SIZE * 2:
                return False
        else:
            # If already split, check if any blob can split
            for blob in self.split_blobs:
                if blob.is_active and blob.size >= PLAYER_MIN_SIZE * 2:
                    return True
            return False
        
        return True
    
    def split(self) -> bool:
        """Split the player into multiple blobs"""
        if not self.can_split():
            return False
        
        if not self.is_split:
            # First split - split main player into 2 blobs
            split_size = self.size / 2
            
            # Ensure minimum size
            if split_size < PLAYER_MIN_SIZE:
                return False
            
            # Create split blobs
            self.split_blobs.clear()
            
            # Create 2 blobs positioned on opposite sides
            for i in range(2):
                # Calculate position offset for each blob
                angle = math.pi * i  # 0 and π (opposite sides)
                offset_distance = self.size * 1.2  # Slightly more than radius
                
                blob = SplitBlob(
                    position=Vector2(0, 0),  # Will be set by set_split_position
                    size=split_size,
                    color=self.color
                )
                
                # Set the blob's split position
                blob.set_split_position(self.position, angle, offset_distance)
                
                self.split_blobs.append(blob)
            
            # Update state
            self.is_split = True
            self.split_count += 1
            self.last_split_time = time.time()
            
        else:
            # Subsequent splits - split each active blob into 2
            new_blobs = []
            
            for blob in self.split_blobs:
                if blob.is_active and blob.size >= PLAYER_MIN_SIZE * 2:
                    # Split this blob into 2
                    split_size = blob.size / 2
                    
                    # Create 2 new blobs
                    for i in range(2):
                        angle = blob.split_angle + (math.pi * i)  # Offset from current angle
                        offset_distance = blob.size * 1.2
                        
                        new_blob = SplitBlob(
                            position=Vector2(0, 0),
                            size=split_size,
                            color=self.color
                        )
                        
                        new_blob.set_split_position(blob.position, angle, offset_distance)
                        new_blobs.append(new_blob)
                    
                    # Deactivate the original blob
                    blob.is_active = False
            
            # Add new blobs to the list
            self.split_blobs.extend(new_blobs)
            self.split_count += 1
        
        return True
    
    def rejoin_blobs(self):
        """Rejoin all split blobs back into the main player"""
        if not self.is_split:
            return
        
        # Calculate total mass from all blobs
        total_mass = 0
        for blob in self.split_blobs:
            if blob.is_active:
                total_mass += blob.size
                blob.deactivate()
        
        # Update main player
        self.size = total_mass
        self.speed = self._calculate_speed()
        
        # Reset split state
        self.is_split = False
        self.split_blobs.clear()
        
        # Update collision properties
        self._update_collision_properties()
    
    def can_eat_food(self, food) -> bool:
        """Check if player can eat the given food"""
        if not self.is_active:
            return False
        
        if self.is_split:
            # Check if any split blob can eat the food
            return any(blob.can_eat_food(food) for blob in self.split_blobs)
        else:
            # Check main player
            return self.size > food.size
    
    def can_eat_enemy(self, enemy) -> bool:
        """Check if player can eat the given enemy"""
        if not self.is_active:
            return False
        
        if self.is_split:
            # Check if any split blob can eat the enemy
            return any(blob.can_eat_enemy(enemy) for blob in self.split_blobs)
        else:
            # Check main player
            return self.size > enemy.size * 1.1
    
    def eat_food(self, food) -> bool:
        """Consume food and grow"""
        if not self.can_eat_food(food):
            return False
        
        if self.is_split:
            # Find the first blob that can eat the food
            for blob in self.split_blobs:
                if blob.can_eat_food(food):
                    return blob.eat_food(food)
            return False
        else:
            # Main player eats food
            self.size += food.score_value
            self.total_mass_gained += food.score_value
            self.speed = self._calculate_speed()
            return True
    
    def eat_enemy(self, enemy) -> bool:
        """Consume enemy and grow"""
        if not self.can_eat_enemy(enemy):
            return False
        
        if self.is_split:
            # Find the first blob that can eat the enemy
            for blob in self.split_blobs:
                if blob.can_eat_enemy(enemy):
                    return blob.eat_enemy(enemy)
            return False
        else:
            # Main player eats enemy
            self.size += enemy.size * 0.8
            self.total_mass_gained += enemy.size * 0.8
            self.speed = self._calculate_speed()
            return True
    
    def check_collision_with_food(self, food) -> bool:
        """Check collision with food (center touch required)"""
        if not self.is_active:
            return False
        
        if self.is_split:
            # Check collision with any split blob
            return any(blob.check_collision_with_food(food) for blob in self.split_blobs)
        else:
            # Check collision with main player
            distance = distance_between_points(self.position, Vector2(food.rect.center))
            return distance <= self.size
    
    def check_collision_with_enemy(self, enemy) -> bool:
        """Check collision with enemy blob"""
        if not self.is_active:
            return False
        
        if self.is_split:
            # Check collision with any split blob
            return any(blob.check_collision_with_enemy(enemy) for blob in self.split_blobs)
        else:
            # Check collision with main player
            return distance_between_points(self.position, enemy.position) <= (self.size + enemy.size)
    
    def draw(self, surface: pygame.Surface, camera_x: float, camera_y: float, 
             zoom_factor: float = 1.0):
        """Draw player with camera offset and zoom scaling"""
        if not self.is_active:
            return
        
        if self.is_split:
            # Draw split blobs
            for blob in self.split_blobs:
                blob.draw(surface, camera_x, camera_y, zoom_factor)
        else:
            # Draw main player
            screen_x = (self.position.x - camera_x) * zoom_factor
            screen_y = (self.position.y - camera_y) * zoom_factor
            
            # Calculate scaled size
            scaled_size = int(self.size * zoom_factor)
            
            # Apply pulse effect
            pulse_factor = 1.0 + 0.05 * math.sin(self.pulse_angle)
            final_size = int(scaled_size * pulse_factor)
            
            # Ensure reasonable size
            final_size = max(2, min(final_size, scaled_size * 2))
            
            # Draw player
            pygame.draw.circle(
                surface,
                self.color,
                (int(screen_x), int(screen_y)),
                final_size
            )
            
            # Draw outline
            pygame.draw.circle(
                surface,
                (255, 255, 255),  # White outline
                (int(screen_x), int(screen_y)),
                final_size,
                2  # 2 pixel outline
            )
    
    def get_total_size(self) -> float:
        """Get total size across all blobs"""
        if self.is_split:
            return sum(blob.size for blob in self.split_blobs if blob.is_active)
        else:
            return self.size
    
    def get_center_position(self) -> Vector2:
        """Get center position for camera following - ALWAYS follows main blob"""
        # Camera should always follow the main blob, not center of mass
        return self.position
    
    def get_zoom_factor(self) -> float:
        """Get zoom factor based on total size"""
        total_size = self.get_total_size()
        return calculate_zoom_factor(total_size)
    
    def __str__(self) -> str:
        """String representation for debugging"""
        status = "split" if self.is_split else "single"
        return f"Player(size={self.size:.1f}, status={status}, blobs={len(self.split_blobs)})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        status = "split" if self.is_split else "single"
        return f"Player(size={self.size:.1f}, status={status}, blobs={len(self.split_blobs)}, pos={self.position})"
