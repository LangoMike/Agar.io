"""
Main game engine that orchestrates all game systems
"""

import pygame
import time
from pygame.math import Vector2
from typing import List, Optional
from utils.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    GAME_STATE_PLAYING,
    GAME_STATE_PAUSED,
    GAME_STATE_GAME_OVER,
    GAME_STATE_VICTORY,
    FOOD_DENSITY,
    FOOD_MULTIPLIER,
    ENEMY_START_COUNT,
)
from entities.player import Player
from entities.food import Food
from entities.enemy import EnemyBlob
from .camera import Camera
from .world import World
from .ui_manager import UIManager
from utils.math_utils import distance_between_points


class GameEngine:
    """Main game engine that manages the game loop and all systems"""

    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Create display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Agar.io - Single Player")

        # Create clock for FPS control
        self.clock = pygame.time.Clock()

        # Game state
        self.game_state = GAME_STATE_PLAYING
        self.running = True
        
        # UI state
        self.show_controls = False

        # Game systems
        self.world = World()
        self.camera = Camera()
        self.ui_manager = UIManager()

        # Game entities
        self.player = Player(self.world.get_center_position())
        self.food_group = pygame.sprite.Group()

        # Game timing
        self.game_start_time = time.time()
        self.game_time = 0
        self.delta_time = 0

        # Food management
        self.max_food = int(self.world.get_area() * FOOD_DENSITY * FOOD_MULTIPLIER)
        self._generate_initial_food()

        # Input handling
        self.mouse_pos = Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.keys_pressed = set()

        # Enemy management
        self.enemies: List[EnemyBlob] = []
        self.max_enemies = 12  # Increased for testing - easier to find enemies
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 5.0
        self._generate_initial_enemies()

    def _generate_initial_food(self):
        """Generate initial food on the map"""
        for _ in range(self.max_food):
            food = Food(self.world.surface)
            self.food_group.add(food)

    def _generate_initial_enemies(self):
        """Generate initial enemies on the map"""
        for _ in range(ENEMY_START_COUNT):  # Use constant instead of max_enemies
            enemy = EnemyBlob(self._get_safe_spawn_position())
            self.enemies.append(enemy)

    def _get_safe_spawn_position(self) -> Vector2:
        """Get a safe spawn position away from player and other entities"""
        max_attempts = 50
        min_distance = 100  # Minimum distance from player and other entities

        for _ in range(max_attempts):
            # Get random position
            position = self.world.get_random_position()

            # Check distance from player
            player_pos = self.player.get_center_position()
            if distance_between_points(position, player_pos) < min_distance:
                continue

            # Check distance from existing enemies
            too_close = False
            for enemy in self.enemies:
                if distance_between_points(position, enemy.position) < min_distance:
                    too_close = True
                    break

            if not too_close:
                return position

        # If we can't find a safe position, return a random one far from player
        return self.world.get_random_position()

    def handle_events(self):
        """Handle all game events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p:
                    self._toggle_pause()
                elif event.key == pygame.K_SPACE:
                    if self.game_state == GAME_STATE_PLAYING:
                        self._handle_split()
                    elif (
                        self.game_state == GAME_STATE_GAME_OVER
                        or self.game_state == GAME_STATE_VICTORY
                    ):
                        self._restart_game()
                elif event.key == pygame.K_c:
                    # Toggle controls display
                    self.show_controls = not self.show_controls

            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = Vector2(event.pos)

            elif event.type == pygame.MOUSEWHEEL:
                # Handle mouse wheel zoom
                self.camera.handle_mouse_wheel(event.y)

            elif event.type == pygame.KEYUP:
                if event.key in self.keys_pressed:
                    self.keys_pressed.remove(event.key)

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.keys_pressed.add("w")
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.keys_pressed.add("s")
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.keys_pressed.add("a")
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.keys_pressed.add("d")

    def _toggle_pause(self):
        """Toggle pause state"""
        if self.game_state == GAME_STATE_PLAYING:
            self.game_state = GAME_STATE_PAUSED
        elif self.game_state == GAME_STATE_PAUSED:
            self.game_state = GAME_STATE_PLAYING

    def _handle_split(self):
        """Handle player split request"""
        if self.player.can_split():
            self.player.split()

    def _restart_game(self):
        """Restart the game"""
        self.__init__()

    def update(self):
        """Update all game systems"""
        if self.game_state != GAME_STATE_PLAYING:
            return

        # Update game time
        self.game_time = time.time() - self.game_start_time

        # Calculate delta time
        self.delta_time = self.clock.get_time() / 1000.0

        # Convert mouse position to world coordinates
        world_mouse_pos = self.camera.screen_to_world(self.mouse_pos)

        # Update player
        self.player.update(self.delta_time, world_mouse_pos)

        # Update camera
        player_center = self.player.get_center_position()
        self.camera.update(player_center, self.delta_time)

        # Update food
        self.food_group.update(self.delta_time)

        # Update enemies
        self._update_enemies()

        # Handle collisions
        self._handle_collisions()

        # Temporarily disabled enemy spawning to prevent overpopulation
        # self._spawn_enemies()

        # Maintain food count
        self._maintain_food_count()

        # Check game over conditions
        self._check_game_over()

        # Check victory conditions
        self._check_victory()

    def _handle_collisions(self):
        """Handle all collision detection and responses"""
        # Check player-food collisions
        food_to_remove = []
        new_food_to_add = []

        for food in self.food_group:
            if self.player.check_collision_with_food(food):
                if self.player.can_eat_food(food):
                    if self.player.eat_food(food):
                        food_to_remove.append(food)
                        # Create new food at random location
                        new_food = Food(self.world.surface)
                        new_food_to_add.append(new_food)

        # Remove eaten food and add new food
        for food in food_to_remove:
            self.food_group.remove(food)

        for food in new_food_to_add:
            self.food_group.add(food)

        # Check player-enemy collisions
        self._handle_player_enemy_collisions()

    def _handle_player_enemy_collisions(self):
        """Handle collisions between player and enemies"""
        player_position = self.player.get_center_position()
        player_size = self.player.get_total_size()

        for enemy in self.enemies[:]:  # Copy list to allow removal
            if not enemy.is_active:
                continue

            # Check if player can eat enemy
            if self.player.can_eat_enemy(enemy):
                if self.player.check_collision_with_enemy(enemy):
                    if self.player.eat_enemy(enemy):
                        # Player ate the enemy
                        self.enemies.remove(enemy)
                        continue

            # Check if enemy can eat player (independent check)
            if enemy.can_eat_player(player_size):
                if enemy.check_collision_with_player(player_position, player_size):
                    # Enemy ate the player - game over!
                    self.game_state = GAME_STATE_GAME_OVER
                    return

    # Check player-enemy collisions
    def _update_enemies(self):
        """Update all enemy AI and movement"""
        # Get current game state for AI decision making
        food_positions = [Vector2(food.rect.center) for food in self.food_group]
        player_position = self.player.get_center_position()
        player_size = self.player.get_total_size()

        # Update each enemy
        for enemy in self.enemies[:]:  # Copy list to allow removal during iteration
            if enemy.is_active:
                # Update enemy AI and movement
                enemy.update(
                    self.delta_time,
                    food_positions,
                    player_position,
                    player_size,
                    self.enemies,
                )

                # Check if enemy should be removed (too small or inactive)
                if enemy.size < 5 or not enemy.is_active:
                    self.enemies.remove(enemy)
                    continue

                # Check enemy-food collisions
                for food in list(self.food_group):  # Convert to list to allow removal
                    if enemy.check_collision_with_food(food):
                        if enemy.can_eat_food(food):
                            if enemy.eat_food(food):
                                self.food_group.remove(food)
                                # Create new food at random location
                                new_food = Food(self.world.surface)
                                self.food_group.add(new_food)

        # Spawn new enemies if needed
        self._spawn_enemies()

    def _spawn_enemies(self):
        """Spawn new enemies to maintain the target count"""
        self.enemy_spawn_timer += self.delta_time

        if (
            self.enemy_spawn_timer >= self.enemy_spawn_rate
            and len(self.enemies) < self.max_enemies
        ):
            # Spawn new enemy
            enemy = EnemyBlob(self._get_safe_spawn_position())
            self.enemies.append(enemy)
            self.enemy_spawn_timer = 0

    def _maintain_food_count(self):
        """Ensure the correct number of food items exist"""
        current_food_count = len(self.food_group)

        if current_food_count < self.max_food:
            # Add more food
            food_to_add = self.max_food - current_food_count
            for _ in range(food_to_add):
                food = Food(self.world.surface)
                self.food_group.add(food)

    def _check_game_over(self):
        """Check if the game should end"""
        # For now, just check if player is too small
        if self.player.get_total_size() < 5:
            self.game_state = GAME_STATE_GAME_OVER

    def _check_victory(self):
        """Check if the player has won (eliminated all enemies)"""
        # Check if all enemies are eliminated
        active_enemies = [enemy for enemy in self.enemies if enemy.is_active]
        if len(active_enemies) == 0:
            self.game_state = GAME_STATE_VICTORY

    def draw(self):
        """Draw all game elements"""
        # Clear screen
        self.screen.fill((0, 0, 0))

        # Draw world background
        self.world.draw(
            self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor
        )

        # Draw food
        for food in self.food_group:
            food.draw(
                self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor
            )

        # Draw player
        self.player.draw(
            self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor
        )
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(
                self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor
            )

        # Draw UI
        player_size = self.player.get_total_size()
        zoom_factor = self.camera.zoom_factor
        split_count = self.player.split_count
        kill_count = self.player.kill_count

        self.ui_manager.draw_ui(
            self.screen,
            player_size,
            self.game_time,
            zoom_factor,
            split_count,
            kill_count,
        )

        # Draw leaderboard
        self.ui_manager.draw_leaderboard(
            self.screen, player_size, self.enemies, self.player.get_center_position()
        )

        # Draw minimap
        food_positions = [Vector2(food.rect.center) for food in self.food_group]
        enemy_positions = [
            enemy.position for enemy in self.enemies if enemy.is_active
        ]  # Will be populated when enemies are added

        self.ui_manager.draw_minimap(
            self.screen,
            self.player.get_center_position(),
            self.camera.x,
            self.camera.y,
            self.camera.zoom_factor,
            food_positions,
            enemy_positions,
        )

        # Draw controls button (controls info shown only when toggled)
        self.ui_manager.draw_controls_button(self.screen)
        
        # Draw controls info only when toggled
        if hasattr(self, 'show_controls') and self.show_controls:
            self.ui_manager.draw_controls_info(self.screen)

        # Draw game state specific screens
        if self.game_state == GAME_STATE_PAUSED:
            self.ui_manager.draw_pause_screen(self.screen)
        elif self.game_state == GAME_STATE_GAME_OVER:
            self.ui_manager.draw_game_over_screen(
                self.screen, player_size, self.game_time
            )
        elif self.game_state == GAME_STATE_VICTORY:
            self.ui_manager.draw_victory_screen(
                self.screen, player_size, self.game_time
            )

        # Update display
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            self.handle_events()

            # Update game
            self.update()

            # Draw everything
            self.draw()

            # Control FPS
            self.clock.tick(FPS)

        # Clean up
        pygame.quit()

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"GameEngine(state={self.game_state}, food={len(self.food_group)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"GameEngine(state={self.game_state}, food={len(self.food_group)}, player_size={self.player.get_total_size():.1f})"
