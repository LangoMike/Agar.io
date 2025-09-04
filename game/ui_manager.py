"""
UI Manager for the Agar.io game
"""

import pygame
from pygame.math import Vector2
from typing import Optional
from utils.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    UI_BG_COLOR,
    UI_TEXT_COLOR,
    UI_BORDER_COLOR,
    MINIMAP_SIZE,
    MINIMAP_OPACITY,
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

    def draw_ui(
        self,
        screen: pygame.Surface,
        player_size: float,
        game_time: float,
        zoom_factor: float,
        split_count: int = 0,
        kill_count: int = 0,
    ):
        """Draw the main UI elements with modern rounded design"""
        # Create smaller UI background with rounded corners
        ui_width = 280
        ui_height = 140
        ui_rect = pygame.Rect(15, 15, ui_width, ui_height)

        # Create rounded rectangle surface
        ui_surface = pygame.Surface((ui_width, ui_height), pygame.SRCALPHA)

        # Draw rounded rectangle background
        corner_radius = 12
        self._draw_rounded_rect(
            ui_surface, (0, 0, ui_width, ui_height), (40, 40, 40, 200), corner_radius
        )

        # Draw border with rounded corners
        self._draw_rounded_rect(
            ui_surface,
            (0, 0, ui_width, ui_height),
            (80, 80, 80, 255),
            corner_radius,
            2,
            True,
        )

        screen.blit(ui_surface, ui_rect)

        # Draw UI text with smaller fonts and better spacing
        y_offset = 25
        x_offset = 25

        # Size/Score (smaller font)
        size_text = self.font_medium.render(
            f"Size: {player_size:.1f}", True, (255, 255, 255)
        )
        screen.blit(size_text, (x_offset, y_offset))
        y_offset += 28

        # Time
        time_text = self.font_small.render(
            f"Time: {game_time:.1f}s", True, (255, 255, 255)
        )
        screen.blit(time_text, (x_offset, y_offset))
        y_offset += 25

        # Zoom factor
        zoom_text = self.font_small.render(
            f"Zoom: {zoom_factor:.2f}x", True, (255, 255, 255)
        )
        screen.blit(zoom_text, (x_offset, y_offset))
        y_offset += 25

        # Split info
        if split_count > 0:
            split_text = self.font_small.render(
                f"Splits: {split_count}", True, (255, 255, 255)
            )
            screen.blit(split_text, (x_offset, y_offset))
            y_offset += 25

        # Kill count
        kill_text = self.font_small.render(
            f"Kills: {kill_count}", True, (255, 255, 255)
        )
        screen.blit(kill_text, (x_offset, y_offset))
        y_offset += 25

        # Growth rate info (show diminishing returns in action)
        if player_size > 20:
            from utils.math_utils import calculate_growth_value

            sample_food_size = 30.0  # Medium food
            actual_growth = calculate_growth_value(sample_food_size, player_size)
            growth_percent = (actual_growth / sample_food_size) * 100
            growth_text = self.font_small.render(
                f"Growth: {growth_percent:.0f}%", True, (200, 255, 200)
            )
            screen.blit(growth_text, (x_offset, y_offset))

    def draw_minimap(
        self,
        screen: pygame.Surface,
        player_position: Vector2,
        camera_x: float,
        camera_y: float,
        zoom_factor: float,
        food_positions: list = None,
        enemy_positions: list = None,
    ):
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
            2,
        )

        # Draw food positions (small green dots)
        if food_positions:
            for food_pos in food_positions:
                minimap_x_pos = int(food_pos.x * scale_x)
                minimap_y_pos = int(food_pos.y * scale_y)
                if (
                    0 <= minimap_x_pos < self.minimap_size
                    and 0 <= minimap_y_pos < self.minimap_size
                ):
                    pygame.draw.circle(
                        self.minimap_surface,
                        (0, 255, 0),  # Green
                        (minimap_x_pos, minimap_y_pos),
                        1,
                    )

        # Draw enemy positions (small red dots)
        if enemy_positions:
            for enemy_pos in enemy_positions:
                minimap_x_pos = int(enemy_pos.x * scale_x)
                minimap_y_pos = int(enemy_pos.y * scale_y)
                if (
                    0 <= minimap_x_pos < self.minimap_size
                    and 0 <= minimap_y_pos < self.minimap_size
                ):
                    pygame.draw.circle(
                        self.minimap_surface,
                        (255, 0, 0),  # Red
                        (minimap_x_pos, minimap_y_pos),
                        1,
                    )

        # Draw player position (red square)
        player_minimap_x = int(player_position.x * scale_x)
        player_minimap_y = int(player_position.y * scale_y)
        if (
            0 <= player_minimap_x < self.minimap_size
            and 0 <= player_minimap_y < self.minimap_size
        ):
            pygame.draw.rect(
                self.minimap_surface,
                (255, 0, 0),  # Red
                (player_minimap_x - 2, player_minimap_y - 2, 4, 4),
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
            min(viewport_height, self.minimap_size - viewport_y),
        )

        if viewport_rect.width > 0 and viewport_rect.height > 0:
            pygame.draw.rect(
                self.minimap_surface,
                (0, 0, 255, 128),  # Semi-transparent blue
                viewport_rect,
                1,
            )

        # Draw minimap to screen
        screen.blit(self.minimap_surface, (minimap_x, minimap_y))

        # Draw minimap border
        pygame.draw.rect(
            screen,
            self.border_color,
            (minimap_x, minimap_y, self.minimap_size, self.minimap_size),
            2,
        )

    def draw_game_over_screen(
        self,
        screen: pygame.Surface,
        final_score: float,
        game_time: float,
        reason: str = "Game Over",
    ):
        """Draw the game over screen"""
        # Create overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        # Draw game over text
        game_over_text = self.font_large.render(reason, True, (255, 255, 255))
        text_rect = game_over_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)
        )
        screen.blit(game_over_text, text_rect)

        # Draw final score
        score_text = self.font_medium.render(
            f"Final Size: {final_score:.1f}", True, (255, 255, 255)
        )
        score_rect = score_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
        )
        screen.blit(score_text, score_rect)

        # Draw game time
        time_text = self.font_medium.render(
            f"Time Survived: {game_time:.1f}s", True, (255, 255, 255)
        )
        time_rect = time_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(time_text, time_rect)

        # Draw restart instruction
        restart_text = self.font_small.render(
            "Press SPACE to restart or ESC to quit", True, (200, 200, 200)
        )
        restart_rect = restart_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
        )
        screen.blit(restart_text, restart_rect)

    def draw_victory_screen(
        self,
        screen: pygame.Surface,
        final_score: float,
        game_time: float,
    ):
        """Draw the victory screen with bright, happy theme"""
        # Create bright overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(100)
        overlay.fill((100, 255, 100))  # Bright green tint
        screen.blit(overlay, (0, 0))

        # Draw victory text with bright colors
        victory_text = self.font_large.render(
            "🎉 VICTORY! 🎉", True, (255, 255, 0)
        )  # Bright yellow
        text_rect = victory_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)
        )
        screen.blit(victory_text, text_rect)

        # Draw final score
        score_text = self.font_medium.render(
            f"Final Size: {final_score:.1f}", True, (255, 255, 255)  # Bright white
        )
        score_rect = score_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)
        )
        screen.blit(score_text, score_rect)

        # Draw game time
        time_text = self.font_medium.render(
            f"Time to Victory: {game_time:.1f}s", True, (255, 255, 255)  # Bright white
        )
        time_rect = time_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(time_text, time_rect)

        # Draw restart instruction
        restart_text = self.font_small.render(
            "Press SPACE to play again or ESC to quit",
            True,
            (200, 255, 200),  # Light green
        )
        restart_rect = restart_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
        )
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
        instruction_text = self.font_medium.render(
            "Press P to resume", True, (200, 200, 200)
        )
        instruction_rect = instruction_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
        )
        screen.blit(instruction_text, instruction_rect)

    def _draw_rounded_rect(
        self, surface, rect, color, radius, border=0, border_only=False
    ):
        """Draw a rounded rectangle with optional border"""
        x, y, width, height = rect

        if border_only:
            # Draw border only
            pygame.draw.rect(surface, color, (x, y, width, height), border)
            # Round the corners by drawing circles at the corners
            pygame.draw.circle(surface, color, (x + radius, y + radius), radius, border)
            pygame.draw.circle(
                surface, color, (x + width - radius, y + radius), radius, border
            )
            pygame.draw.circle(
                surface, color, (x + radius, y + height - radius), radius, border
            )
            pygame.draw.circle(
                surface,
                color,
                (x + width - radius, y + height - radius),
                radius,
                border,
            )
        else:
            # Draw filled rounded rectangle
            # Main rectangle
            pygame.draw.rect(
                surface, color, (x + radius, y, width - 2 * radius, height)
            )
            pygame.draw.rect(
                surface, color, (x, y + radius, width, height - 2 * radius)
            )

            # Corner circles
            pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
            pygame.draw.circle(surface, color, (x + width - radius, y + radius), radius)
            pygame.draw.circle(
                surface, color, (x + radius, y + height - radius), radius
            )
            pygame.draw.circle(
                surface, color, (x + width - radius, y + height - radius), radius
            )

    def draw_leaderboard(
        self,
        screen: pygame.Surface,
        player_size: float,
        enemies: list,
        player_position: Vector2,
    ):
        """Draw the leaderboard showing top 10 blobs by size"""
        # Leaderboard dimensions and position (smaller width, expanded height)
        lb_width = 280
        lb_height = 500  # Expanded to fit all players
        lb_x = SCREEN_WIDTH - lb_width - 15
        lb_y = 15

        # Create leaderboard surface with rounded corners
        lb_surface = pygame.Surface((lb_width, lb_height), pygame.SRCALPHA)

        # Draw dark grey background with rounded corners
        corner_radius = 12
        self._draw_rounded_rect(
            lb_surface, (0, 0, lb_width, lb_height), (30, 30, 30, 220), corner_radius
        )

        # Draw border
        self._draw_rounded_rect(
            lb_surface, lb_surface.get_rect(), (80, 80, 80, 255), corner_radius, 2, True
        )

        screen.blit(lb_surface, (lb_x, lb_y))

        # Title (smaller font)
        title_text = self.font_medium.render("Leaderboard", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(lb_x + lb_width // 2, lb_y + 25))
        screen.blit(title_text, title_rect)

        # Collect all blobs and sort by size
        all_blobs = []

        # Add player
        all_blobs.append(
            {
                "name": "You",
                "size": player_size,
                "position": player_position,
                "is_player": True,
            }
        )

        # Add enemies
        for i, enemy in enumerate(enemies):
            if enemy.is_active:
                all_blobs.append(
                    {
                        "name": f"Enemy {i + 1}",
                        "size": enemy.size,
                        "position": enemy.position,
                        "is_player": False,
                    }
                )

        # Sort by size (largest first)
        all_blobs.sort(key=lambda x: x["size"], reverse=True)

        # Display top 10 with borders around each entry
        y_offset = 60
        entry_height = 30

        for i, blob in enumerate(all_blobs[:10]):
            # Draw border around this entry
            entry_rect = pygame.Rect(
                lb_x + 10, lb_y + y_offset - 5, lb_width - 20, entry_height
            )
            pygame.draw.rect(
                screen, (100, 100, 100, 100), entry_rect, 1
            )  # Light grey border

            # Position indicator
            pos_text = self.font_small.render(f"{i + 1}.", True, (200, 200, 200))
            screen.blit(pos_text, (lb_x + 20, lb_y + y_offset))

            # Name (yellow for player, white for others)
            name_color = (255, 255, 0) if blob["is_player"] else (255, 255, 255)
            name_text = self.font_small.render(blob["name"], True, name_color)
            screen.blit(name_text, (lb_x + 50, lb_y + y_offset))

            # Size (score) - right-aligned within the leaderboard box
            score_text = self.font_small.render(
                f"{blob['size']:.0f}", True, (200, 200, 200)
            )
            # Position score text at the right side with padding (relative to leaderboard)
            score_x = lb_x + lb_width - 20 - score_text.get_width()
            screen.blit(score_text, (score_x, lb_y + y_offset))

            y_offset += entry_height + 5  # Spacing between bordered entries

    def draw_controls_button(self, screen: pygame.Surface):
        """Draw a controls toggle button below the leaderboard"""
        # Position controls button below the leaderboard
        lb_width = 280
        lb_x = SCREEN_WIDTH - lb_width - 15
        lb_y = 15
        lb_height = 500

        button_y = lb_y + lb_height + 20
        button_width = 200
        button_height = 35
        # Center the button with the leaderboard
        button_x = lb_x + (lb_width - button_width) // 2

        # Create button surface with rounded corners
        button_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)

        # Draw translucent background
        pygame.draw.rect(
            button_surface, (60, 60, 60, 150), (0, 0, button_width, button_height)
        )

        # Draw border
        pygame.draw.rect(
            button_surface, (120, 120, 120, 200), (0, 0, button_width, button_height), 2
        )

        # Round the corners
        corner_radius = 8
        self._draw_rounded_rect(
            button_surface,
            (0, 0, button_width, button_height),
            (60, 60, 60, 150),
            corner_radius,
        )
        self._draw_rounded_rect(
            button_surface,
            (0, 0, button_width, button_height),
            (120, 120, 120, 200),
            corner_radius,
            2,
            True,
        )

        # Draw button text
        button_font = pygame.font.Font(None, self.font_small.get_height())
        button_text = button_font.render(
            'Press "C" to view controls', True, (255, 255, 255)
        )
        text_rect = button_text.get_rect(center=(button_width // 2, button_height // 2))
        button_surface.blit(button_text, text_rect)

        # Draw button on screen
        screen.blit(button_surface, (button_x, button_y))

    def draw_controls_info(self, screen: pygame.Surface):
        """Draw controls information below the leaderboard (when toggled)"""
        # Position controls below the leaderboard
        lb_width = 280
        lb_x = SCREEN_WIDTH - lb_width - 15
        lb_y = 15
        lb_height = 500

        y_offset = lb_y + lb_height + 70  # Below the button
        x_offset = lb_x

        # Create controls panel with rounded corners
        panel_width = 250
        panel_height = 150
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)

        # Draw translucent background
        self._draw_rounded_rect(
            panel_surface, (0, 0, panel_width, panel_height), (60, 60, 60, 150), 10
        )

        # Draw border
        self._draw_rounded_rect(
            panel_surface,
            (0, 0, panel_width, panel_height),
            (120, 120, 120, 200),
            10,
            2,
            True,
        )

        # Title
        title_text = self.font_medium.render("Controls:", True, (255, 255, 255))
        title_x = (panel_width - title_text.get_width()) // 2
        panel_surface.blit(title_text, (title_x, 15))

        # Controls list
        controls = [
            "-Mouse Direction = Move",
            "-Mouse Wheel = Zoom",
            "-SPACE = Split",
            "-P = Pause",
            "-ESC = Quit",
        ]

        y_offset = 45
        for control in controls:
            text = self.font_small.render(control, True, (200, 200, 200))
            text_x = (panel_width - text.get_width()) // 2
            panel_surface.blit(text, (text_x, y_offset))
            y_offset += 20

        # Draw panel on screen
        screen.blit(panel_surface, (x_offset, lb_y + lb_height + 70))

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"UIManager(minimap={self.minimap_size})"

    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return f"UIManager(minimap={self.minimap_size}, opacity={self.minimap_opacity})"
