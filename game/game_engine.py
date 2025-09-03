"""
Main game engine that orchestrates all game systems
"""

import pygame
import time
from pygame.math import Vector2
from typing import List, Optional
from utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GAME_STATE_PLAYING,
    GAME_STATE_PAUSED, GAME_STATE_GAME_OVER, FOOD_DENSITY, FOOD_MULTIPLIER
)
from entities.player import Player
from entities.food import Food
from .camera import Camera
from .world import World
from .ui_manager import UIManager

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
        
    def _generate_initial_food(self):
        """Generate initial food on the map"""
        for _ in range(self.max_food):
            food = Food(self.world.surface)
            self.food_group.add(food)
    
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
                    elif self.game_state == GAME_STATE_GAME_OVER:
                        self._restart_game()
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = Vector2(event.pos)
            
            elif event.type == pygame.KEYUP:
                if event.key in self.keys_pressed:
                    self.keys_pressed.remove(event.key)
        
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.keys_pressed.add('w')
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.keys_pressed.add('s')
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.keys_pressed.add('a')
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.keys_pressed.add('d')
    
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
        target_zoom = self.player.get_zoom_factor()
        self.camera.update(player_center, target_zoom, self.delta_time)
        
        # Update food
        self.food_group.update(self.delta_time)
        
        # Handle collisions
        self._handle_collisions()
        
        # Maintain food count
        self._maintain_food_count()
        
        # Check game over conditions
        self._check_game_over()
    
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
    
    def draw(self):
        """Draw all game elements"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw world background
        self.world.draw(self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor)
        
        # Draw food
        for food in self.food_group:
            food.draw(self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor)
        
        # Draw player
        self.player.draw(self.screen, self.camera.x, self.camera.y, self.camera.zoom_factor)
        
        # Draw UI
        player_size = self.player.get_total_size()
        zoom_factor = self.camera.zoom_factor
        split_count = self.player.split_count
        
        self.ui_manager.draw_ui(
            self.screen, player_size, self.game_time, zoom_factor, split_count
        )
        
        # Draw minimap
        food_positions = [Vector2(food.rect.center) for food in self.food_group]
        enemy_positions = []  # Will be populated when enemies are added
        
        self.ui_manager.draw_minimap(
            self.screen, self.player.get_center_position(),
            self.camera.x, self.camera.y, self.camera.zoom_factor,
            food_positions, enemy_positions
        )
        
        # Draw controls info
        self.ui_manager.draw_controls_info(self.screen)
        
        # Draw game state specific screens
        if self.game_state == GAME_STATE_PAUSED:
            self.ui_manager.draw_pause_screen(self.screen)
        elif self.game_state == GAME_STATE_GAME_OVER:
            self.ui_manager.draw_game_over_screen(
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
