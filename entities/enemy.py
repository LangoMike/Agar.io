"""
Enemy blob class that will be controlled by AI
"""

import pygame
import math
import random
from pygame.math import Vector2
from typing import List, Optional, Tuple
from utils.constants import (
    ENEMY_MIN_SIZE,
    ENEMY_MAX_SIZE,
    ENEMY_BASE_SPEED,
    ENEMY_COLORS,
    WORLD_WIDTH,
    WORLD_HEIGHT,
)
from utils.math_utils import (
    distance_between_points,
    normalize_vector,
    clamp_value,
    calculate_growth_value,
)


class EnemyBlob:
    """AI-controlled enemy blob that competes with the player"""

    def __init__(self, start_position: Vector2, start_size: float = None):
        # Position and size
        self.position = Vector2(start_position)
        self.size = (
            start_size
            if start_size
            else ENEMY_MIN_SIZE  # Always start at minimum size (20)
        )
        self.original_size = self.size

        # Visual properties
        self.color = random.choice(ENEMY_COLORS)
        self.pulse_angle = random.uniform(0, 2 * math.pi)  # Random starting pulse
        self.pulse_speed = 0.1

        # Movement properties
        self.velocity = Vector2(0, 0)
        self.target_position = Vector2(start_position)
        self.speed = self._calculate_speed()

        # AI properties
        self.ai_state = "seeking"  # seeking, fleeing, hunting, idle
        self.ai_timer = 0
        self.ai_update_rate = 0.1  # Update AI every 100ms
        self.last_ai_decision = 0

        # Collision properties
        self.collision_radius = self.size
        self.collision_rect = pygame.Rect(
            start_position.x - self.size,
            start_position.y - self.size,
            self.size * 2,
            self.size * 2,
        )

        # State properties
        self.is_active = True
        self.total_mass_gained = 0
        self.survival_time = 0

        # AI memory (for learning)
        self.successful_moves = []
        self.failed_moves = []
        self.last_positions = []  # Track recent movement history

    def _calculate_speed(self) -> float:
        """Calculate movement speed based on size (larger = slower)"""
        # Similar to player but with enemy-specific values
        speed_power = 0.2
        return ENEMY_BASE_SPEED / max(1, self.size**speed_power)

    def update(
        self,
        dt: float,
        food_positions: List[Vector2],
        player_position: Vector2,
        player_size: float,
        other_enemies: List = None,
    ):
        """Update enemy position and AI behavior"""
        if not self.is_active:
            return

        # Update survival time
        self.survival_time += dt

        # Update AI behavior
        self._update_ai_behavior(
            dt, food_positions, player_position, player_size, other_enemies
        )

        # Update movement
        self._update_movement(dt)

        # Update collision properties
        self._update_collision_properties()

        # Update visual effects
        self._update_visual_effects(dt)

        # Update AI memory
        self._update_ai_memory()

    def _update_ai_behavior(
        self,
        dt: float,
        food_positions: List[Vector2],
        player_position: Vector2,
        player_size: float,
        other_enemies: List = None,
    ):
        """Update AI decision making"""
        self.ai_timer += dt

        # Only make AI decisions at specified intervals
        if self.ai_timer >= self.ai_update_rate:
            self.ai_timer = 0
            self._make_ai_decision(
                food_positions, player_position, player_size, other_enemies
            )

    def _make_ai_decision(
        self,
        food_positions: List[Vector2],
        player_position: Vector2,
        player_size: float,
        other_enemies: List = None,
    ):
        """Make a decision about what to do next"""
        # Calculate distances and threats
        player_distance = distance_between_points(self.position, player_position)
        player_threat = self._calculate_player_threat(player_distance, player_size)

        # Find nearest food
        nearest_food = self._find_nearest_food(food_positions)
        food_distance = (
            distance_between_points(self.position, nearest_food)
            if nearest_food
            else float("inf")
        )

        # Decision making logic
        if player_threat > 0.7:  # High threat - run away!
            self.ai_state = "fleeing"
            self.target_position = self._calculate_escape_position(player_position)

        elif (
            food_distance < 100 and self.size < 100
        ):  # Close to food and small - eat it!
            self.ai_state = "hunting"
            self.target_position = nearest_food

        elif self.size > player_size * 1.2:  # Bigger than player - hunt them!
            self.ai_state = "hunting"
            self.target_position = player_position

        else:  # Default behavior - seek food
            self.ai_state = "seeking"
            if nearest_food:
                self.target_position = nearest_food
            else:
                # Random movement if no food nearby
                self.target_position = self._get_random_target()

    def _calculate_player_threat(self, distance: float, player_size: float) -> float:
        """Calculate how threatening the player is (0.0 = safe, 1.0 = deadly)"""
        if player_size <= self.size:
            return 0.0  # Player is smaller, not a threat

        # Threat increases as player gets bigger and closer
        size_threat = (player_size - self.size) / max(1, self.size)
        distance_threat = max(0, (200 - distance) / 200)  # Closer = more threatening

        return min(1.0, (size_threat + distance_threat) / 2)

    def _find_nearest_food(self, food_positions: List[Vector2]) -> Optional[Vector2]:
        """Find the closest food to the enemy"""
        if not food_positions:
            return None

        nearest_food = None
        min_distance = float("inf")

        for food_pos in food_positions:
            distance = distance_between_points(self.position, food_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_food = food_pos

        return nearest_food

    def _calculate_escape_position(self, threat_position: Vector2) -> Vector2:
        """Calculate a position to escape to when threatened"""
        # Move away from threat in opposite direction
        escape_direction = self.position - threat_position
        if escape_direction.length() > 0:
            escape_direction = normalize_vector(escape_direction)
            escape_distance = 200  # Escape distance
            escape_position = self.position + (escape_direction * escape_distance)

            # Ensure escape position is within world bounds
            escape_position.x = clamp_value(
                escape_position.x, self.size, WORLD_WIDTH - self.size
            )
            escape_position.y = clamp_value(
                escape_position.y, self.size, WORLD_HEIGHT - self.size
            )

            return escape_position
        else:
            # If we're exactly at threat position, move randomly
            return self._get_random_target()

    def _get_random_target(self) -> Vector2:
        """Get a random target position within world bounds"""
        margin = 100  # Stay away from edges
        x = random.uniform(margin, WORLD_WIDTH - margin)
        y = random.uniform(margin, WORLD_HEIGHT - margin)
        return Vector2(x, y)

    def _update_movement(self, dt: float):
        """Update enemy movement toward target"""
        # Calculate direction to target
        direction = self.target_position - self.position

        if direction.length() > 5:  # Only move if we're not at target
            direction = normalize_vector(direction)

            # Move toward target
            movement = direction * self.speed * dt
            self.position += movement

            # Clamp to world boundaries
            self.position.x = clamp_value(
                self.position.x, self.size, WORLD_WIDTH - self.size
            )
            self.position.y = clamp_value(
                self.position.y, self.size, WORLD_HEIGHT - self.size
            )

    def _update_collision_properties(self):
        """Update collision rectangle and radius based on current position"""
        self.collision_radius = self.size
        self.collision_rect.center = self.position
        self.collision_rect.width = self.size * 2
        self.collision_rect.height = self.size * 2

    def _update_visual_effects(self, dt: float):
        """Update visual effects like pulse animation"""
        self.pulse_angle += self.pulse_speed * dt
        if self.pulse_angle > 2 * math.pi:
            self.pulse_angle = 0

    def _update_ai_memory(self):
        """Update AI memory for learning purposes"""
        # Store recent positions (last 10 positions)
        self.last_positions.append(Vector2(self.position))
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)

    def can_eat_food(self, food) -> bool:
        """Check if enemy can eat the given food"""
        return self.size > food.size

    def can_eat_player(self, player_size: float) -> bool:
        """Check if enemy can eat the player"""
        # Enemy must be significantly larger to eat player (prevents immediate game over)
        return self.size > player_size * 1.5

    def eat_food(self, food) -> bool:
        """Consume food and grow with diminishing returns"""
        if not self.can_eat_food(food):
            return False

        # Grow and update speed with diminishing returns
        growth_value = calculate_growth_value(food.size, self.size)
        self.size += growth_value
        self.total_mass_gained += growth_value
        self.speed = self._calculate_speed()

        # Record successful move for learning
        self.successful_moves.append(
            {
                "action": "eat_food",
                "position": Vector2(self.position),
                "result": "success",
            }
        )

        return True

    def check_collision_with_food(self, food) -> bool:
        """Check collision with food (center touch required)"""
        if not self.is_active:
            return False

        distance = distance_between_points(self.position, Vector2(food.rect.center))
        return distance <= self.size

    def check_collision_with_player(
        self, player_position: Vector2, player_size: float
    ) -> bool:
        """Check collision with player (center touch required)"""
        if not self.is_active:
            return False

        distance = distance_between_points(self.position, player_position)
        # Collision occurs when distance <= sum of both radii
        return distance <= (self.size + player_size)

    def draw(
        self,
        surface: pygame.Surface,
        camera_x: float,
        camera_y: float,
        zoom_factor: float = 1.0,
    ):
        """Draw enemy with camera offset and zoom scaling"""
        if not self.is_active:
            return

        # Convert world position to screen position
        screen_x = (self.position.x - camera_x) * zoom_factor
        screen_y = (self.position.y - camera_y) * zoom_factor

        # Calculate scaled size
        scaled_size = int(self.size * zoom_factor)

        # Apply pulse effect
        pulse_factor = 1.0 + 0.05 * math.sin(self.pulse_angle)
        final_size = int(scaled_size * pulse_factor)

        # Ensure reasonable size
        final_size = max(2, min(final_size, scaled_size * 2))

        # Draw enemy
        pygame.draw.circle(
            surface, self.color, (int(screen_x), int(screen_y)), final_size
        )

        # Draw outline
        pygame.draw.circle(
            surface,
            (255, 255, 255),  # White outline
            (int(screen_x), int(screen_y)),
            final_size,
            2,  # 2 pixel outline
        )

        # Draw AI state indicator (small colored dot)
        state_colors = {
            "seeking": (0, 255, 0),  # Green
            "hunting": (255, 0, 0),  # Red
            "fleeing": (255, 255, 0),  # Yellow
            "idle": (128, 128, 128),  # Gray
        }

        state_color = state_colors.get(self.ai_state, (128, 128, 128))
        indicator_size = max(2, final_size // 4)
        pygame.draw.circle(
            surface, state_color, (int(screen_x), int(screen_y)), indicator_size
        )

    def deactivate(self):
        """Deactivate the enemy"""
        self.is_active = False

    def get_score(self) -> float:
        """Calculate enemy score based on performance"""
        # Score based on size gained and survival time
        size_score = self.size * 0.5
        survival_score = self.survival_time * 0.1
        return size_score + survival_score

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"EnemyBlob(size={self.size:.1f}, state={self.ai_state}, pos={self.position})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"EnemyBlob(size={self.size:.1f}, state={self.ai_state}, pos={self.position}, active={self.is_active})"
