"""
UI Manager for the Agar.io game
"""

import pygame
from pygame.math import Vector2
from typing import Optional
from utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, UI_BG_COLOR, UI_TEXT_COLOR, 
    UI_BORDER_COLOR, MINIMAP_SIZE, MINIMAP_OPACITY
)

class UIManager:
    """Manages all UI elements including minimap, score, and game info"""
    
    def __init__(self):
        # Font setup
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # UI colors
        self.bg_color = UI_BG_COLOR
        self.text_color = UI_TEXT_COLOR
        self.border_color = UI_BORDER_COLOR
        
        # Minimap settings
        self.minimap_size = MINIMAP_SIZE
        self.minimap_opacity = MINIMAP_OPACITY
        
        # Create minimap surface
        self.minimap_surface = pygame.Surface((self.minimap_size, self.minimap_size))
        self.minimap_surface.set_alpha(self.minimap_opacity)
        
    def draw_ui(self, screen: pygame.Surface, player_size: float, 
                game_time: float, zoom_factor: float, split_count: int = 0):
        """Draw the main UI elements"""
        # Create UI background
        ui_rect = pygame.Rect(10, 10, 300, 120)
        
        # Draw semi-transparent background
        ui_surface = pygame.Surface((ui_rect.width, ui_rect.height))
        ui_surface.set_alpha(180)
        ui_surface.fill(self.bg_color)
        screen.blit(ui_surface, ui_rect)
        
        # Draw border
        pygame.draw.rect(screen, self.border_color, ui_rect, 2)
        
        # Draw UI text
        y_offset = 20
        
        # Size/Score
        size_text = self.font_large.render(f"Size: {player_size:.1f}", True, self.text_color)
        screen.blit(size_text, (20, y_offset))
        y_offset += 35
        
        # Time
        time_text = self.font_medium.render(f"Time: {game_time:.1f}s", True, self.text_color)
        screen.blit(time_text, (20, y_offset))
        y_offset += 30
        
        # Zoom factor
        zoom_text = self.font_medium.render(f"Zoom: {zoom_factor:.2f}x", True, self.text_color)
        screen.blit(zoom_text, (20, y_offset))
        y_offset += 30
        
        # Split info
        if split_count > 0:
            split_text = self.font_medium.render(f"Splits: {split_count}", True, self.text_color)
            screen.blit(split_text, (20, y_offset))
    
    def draw_minimap(self, screen: pygame.Surface, player_position: Vector2, 
                     camera_x: float, camera_y: float, zoom_factor: float,
                     food_positions: list = None, enemy_positions: list = None):
        """Draw the minimap in the bottom-left corner"""
        # Calculate minimap position
        minimap_x = 10
        minimap_y = SCREEN_HEIGHT - self.minimap_size - 10
        
        # Clear minimap surface
        self.minimap_surface.fill((255, 255, 255, 0))
        
        # Calculate scale factor (world to minimap)
        world_width = 19200 * 1.2  # From constants
        world_height = 10800 * 1.2
        scale_x = self.minimap_size / world_width
        scale_y = self.minimap_size / world_height
        
        # Draw world border
        pygame.draw.rect(
            self.minimap_surface,
            (100, 100, 100),
            (0, 0, self.minimap_size, self.minimap_size),
            2
        )
        
        # Draw food positions (small green dots)
        if food_positions:
            for food_pos in food_positions:
                minimap_x_pos = int(food_pos.x * scale_x)
                minimap_y_pos = int(food_pos.y * scale_y)
                if 0 <= minimap_x_pos < self.minimap_size and 0 <= minimap_y_pos < self.minimap_size:
                    pygame.draw.circle(
                        self.minimap_surface,
                        (0, 255, 0),  # Green
                        (minimap_x_pos, minimap_y_pos),
                        1
                    )
        
        # Draw enemy positions (small red dots)
        if enemy_positions:
            for enemy_pos in enemy_positions:
                minimap_x_pos = int(enemy_pos.x * scale_x)
                minimap_y_pos = int(enemy_pos.y * scale_y)
                if 0 <= minimap_x_pos < self.minimap_size and 0 <= minimap_y_pos < self.minimap_size:
                    pygame.draw.circle(
                        self.minimap_surface,
                        (255, 0, 0),  # Red
                        (minimap_x_pos, minimap_y_pos),
                        1
                    )
        
        # Draw player position (red square)
        player_minimap_x = int(player_position.x * scale_x)
        player_minimap_y = int(player_position.y * scale_y)
        if 0 <= player_minimap_x < self.minimap_size and 0 <= player_minimap_y < self.minimap_size:
            pygame.draw.rect(
                self.minimap_surface,
                (255, 0, 0),  # Red
                (player_minimap_x - 2, player_minimap_y - 2, 4, 4)
            )
        
        # Draw camera viewport (blue rectangle)
        viewport_width = int((SCREEN_WIDTH / zoom_factor) * scale_x)
        viewport_height = int((SCREEN_HEIGHT / zoom_factor) * scale_y)
        viewport_x = int(camera_x * scale_x)
        viewport_y = int(camera_y * scale_y)
        
        # Ensure viewport rectangle is within minimap bounds
        viewport_rect = pygame.Rect(
            max(0, viewport_x),
            max(0, viewport_y),
            min(viewport_width, self.minimap_size - viewport_x),
            min(viewport_height, self.minimap_size - viewport_y)
        )
        
        if viewport_rect.width > 0 and viewport_rect.height > 0:
            pygame.draw.rect(
                self.minimap_surface,
                (0, 0, 255, 128),  # Semi-transparent blue
                viewport_rect,
                1
            )
        
        # Draw minimap to screen
        screen.blit(self.minimap_surface, (minimap_x, minimap_y))
        
        # Draw minimap border
        pygame.draw.rect(
            screen,
            self.border_color,
            (minimap_x, minimap_y, self.minimap_size, self.minimap_size),
            2
        )
    
    def draw_game_over_screen(self, screen: pygame.Surface, final_score: float, 
                             game_time: float, reason: str = "Game Over"):
        """Draw the game over screen"""
        # Create overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Draw game over text
        game_over_text = self.font_large.render(reason, True, (255, 255, 255))
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))
        screen.blit(game_over_text, text_rect)
        
        # Draw final score
        score_text = self.font_medium.render(f"Final Size: {final_score:.1f}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        screen.blit(score_text, score_rect)
        
        # Draw game time
        time_text = self.font_medium.render(f"Time Survived: {game_time:.1f}s", True, (255, 255, 255))
        time_rect = time_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(time_text, time_rect)
        
        # Draw restart instruction
        restart_text = self.font_small.render("Press SPACE to restart or ESC to quit", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        screen.blit(restart_text, restart_rect)
    
    def draw_pause_screen(self, screen: pygame.Surface):
        """Draw the pause screen"""
        # Create overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Draw pause text
        pause_text = self.font_large.render("PAUSED", True, (255, 255, 255))
        text_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(pause_text, text_rect)
        
        # Draw instruction
        instruction_text = self.font_medium.render("Press P to resume", True, (200, 200, 200))
        instruction_rect = instruction_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        screen.blit(instruction_text, instruction_rect)
    
    def draw_controls_info(self, screen: pygame.Surface):
        """Draw controls information"""
        controls = [
            "Controls:",
            "WASD / Mouse - Move",
            "SPACE - Split",
            "P - Pause",
            "ESC - Quit"
        ]
        
        y_offset = SCREEN_HEIGHT - 150
        
        for i, control in enumerate(controls):
            if i == 0:
                text = self.font_medium.render(control, True, self.text_color)
            else:
                text = self.font_small.render(control, True, self.text_color)
            
            screen.blit(text, (SCREEN_WIDTH - 200, y_offset))
            y_offset += 25
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return f"UIManager(minimap={self.minimap_size})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"UIManager(minimap={self.minimap_size}, opacity={self.minimap_opacity})"
