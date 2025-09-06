"""
AI Training System for Agar.io Enemies
Implements automated AI vs AI training with reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import shutil
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json
import os
from datetime import datetime

# Import our custom modules
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_training.data_collector import DataCollector, TrainingEpisode, calculate_reward
from ai_training.neural_network import (
    AgarAINetwork,
    StateProcessor,
    ModelManager,
    create_beginner_model,
    create_intermediate_model,
    create_hard_model,
)
from entities.enemy import EnemyBlob
from entities.food import Food
from game.world import World
from utils.constants import *
from pygame.math import Vector2


class AITrainer:
    """Main training system for AI enemies"""

    def __init__(
        self,
        difficulty: str = "beginner",
        num_enemies: int = 20,
        training_episodes: int = 100,
    ):
        self.difficulty = difficulty
        self.num_enemies = num_enemies
        self.training_episodes = training_episodes

        # Initialize components
        self.data_collector = DataCollector()
        self.state_processor = StateProcessor()
        self.model_manager = ModelManager()

        # Create neural network model
        if difficulty == "beginner":
            # Try to load the best performing model (episode 27) as starting point
            best_model_path = "ai_training/models/beginner_model_episode_27.pth"
            if os.path.exists(best_model_path):
                print(f"Loading best model: {best_model_path}")
                self.model, metadata = self.model_manager.load_model(best_model_path)
                print(
                    f"   Loaded model from episode {metadata.get('episode', 'unknown')}"
                )
                print(
                    f"   Previous performance: {metadata.get('performance_metrics', {}).get('performance', 'unknown')}"
                )
            else:
                print("Creating new beginner model")
                self.model = create_beginner_model()
        elif difficulty == "intermediate":
            self.model = create_intermediate_model()
        else:  # hard
            self.model = create_hard_model()

        # Move model to GPU if available (using DirectML for AMD GPUs on Windows)
        try:
            import torch_directml

            self.device = torch_directml.device()
            print(f"Using DirectML device: {self.device}")
        except ImportError:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)

        # Improved training parameters for faster learning
        self.learning_rate = 0.001  # Increased from default
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Training progress tracking
        self.best_performance = 0.0
        self.best_model_path = None

        # Game simulation parameters - optimized for faster training
        if difficulty == "beginner":
            self.max_episode_time = 25.0  # Slightly increased for better learning
        elif difficulty == "intermediate":
            self.max_episode_time = 30.0  # Medium complexity
        else:  # hard
            self.max_episode_time = 40.0  # Full complexity

        self.dt = 1.0 / 25.0  # Increased to 25 FPS for better simulation quality

        print(f"AI Trainer initialized for {difficulty} difficulty")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"   Training episodes: {training_episodes}")
        print(f"   Enemies per episode: {num_enemies}")

    def create_training_environment(self) -> Tuple[List[EnemyBlob], List[Food], World]:
        """Create a training environment with enemies and food"""
        # Create world
        world = World()

        # Create enemies with varied starting positions for better training diversity
        enemies = []
        for i in range(self.num_enemies):
            # Vary spawn positions - some near center, some near edges for diverse scenarios
            if i < self.num_enemies // 2:
                # Center area spawns
                x = random.uniform(WORLD_WIDTH * 0.3, WORLD_WIDTH * 0.7)
                y = random.uniform(WORLD_HEIGHT * 0.3, WORLD_HEIGHT * 0.7)
            else:
                # Edge area spawns for more challenging scenarios
                x = random.uniform(50, WORLD_WIDTH - 50)
                y = random.uniform(50, WORLD_HEIGHT - 50)

            position = Vector2(x, y)

            enemy = EnemyBlob(position, ENEMY_MIN_SIZE)
            # Add milestone tracking to prevent spam
            enemy.milestones_achieved = set()
            # Reset exploration tracking for each episode
            enemy.visited_positions = set()
            enemy.total_movement = 0.0
            # Reset wall avoidance tracking
            enemy.wall_history = []
            enemy.wall_warning_shown = False
            enemy.corner_warning_shown = False
            enemy.wall_riding_warning_shown = False
            enemies.append(enemy)

        # Create food
        food_group = []
        num_food = int(WORLD_WIDTH * WORLD_HEIGHT * FOOD_DENSITY)
        for _ in range(num_food):
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            food = Food(world_surface=None, center=(x, y))
            food_group.append(food)

        return enemies, food_group, world

    def simulate_episode(self) -> TrainingEpisode:
        """Simulate a single training episode"""
        # Create environment
        enemies, food_group, world = self.create_training_environment()

        # Start data collection
        episode_id = self.data_collector.start_episode()

        # Episode state
        episode_time = 0.0
        active_enemies = [enemy for enemy in enemies if enemy.is_active]

        # Main simulation loop - optimized for batch processing
        # Enhanced early termination conditions for beginner AI
        min_survival_time = 2.0  # Reduced for faster feedback
        target_size_increase = 200.0  # Adjusted goal for faster episodes
        max_time_for_target = 15.0  # Adjusted for faster episodes

        while episode_time < self.max_episode_time and len(active_enemies) > 0:
            # Process all active enemies in batch for better performance
            batch_states = []
            batch_enemies = []

            # Collect states for all active enemies
            for enemy in active_enemies:
                if not enemy.is_active:
                    continue

                state_tensor = self._get_enemy_state(
                    enemy, enemies, food_group, episode_time
                )
                batch_states.append(state_tensor)
                batch_enemies.append(enemy)

            if not batch_states:
                break

            # Process all enemies in a single batch (much faster on GPU)
            batch_states_tensor = torch.stack(batch_states).to(self.device)

            # Get actions for all enemies at once
            with torch.no_grad():
                batch_outputs = self.model(batch_states_tensor)

                # Split outputs into movement and state components
                movement_logits = batch_outputs[:, :8]
                state_logits = batch_outputs[:, 8:]

                # Get actions for each enemy
                movement_probs = F.softmax(movement_logits, dim=1)
                state_probs = F.softmax(state_logits, dim=1)

                movement_actions = torch.multinomial(movement_probs, 1).cpu().numpy()
                state_actions = torch.multinomial(state_probs, 1).cpu().numpy()
                confidence_scores = torch.max(movement_probs, dim=1)[0].cpu().numpy()

            # Process each enemy with their actions
            for i, enemy in enumerate(batch_enemies):
                movement_dir = movement_actions[i][0]
                state_transition = state_actions[i][0]
                confidence = confidence_scores[i]

                # Record state and action
                self.data_collector.record_state(
                    enemy_position=(enemy.position.x, enemy.position.y),
                    enemy_size=enemy.size,
                    enemy_velocity=(enemy.velocity.x, enemy.velocity.y),
                    player_position=(
                        WORLD_WIDTH / 2,
                        WORLD_HEIGHT / 2,
                    ),  # Simulated player
                    player_size=50.0,  # Simulated player size
                    food_positions=[
                        (food.rect.centerx, food.rect.centery)
                        for food in food_group[:5]
                    ],
                    food_sizes=[food.size for food in food_group[:5]],
                    other_enemy_positions=[
                        (e.position.x, e.position.y)
                        for e in enemies
                        if e != enemy and e.is_active
                    ][:3],
                    other_enemy_sizes=[
                        e.size for e in enemies if e != enemy and e.is_active
                    ][:3],
                    game_time=episode_time,
                    world_width=WORLD_WIDTH,
                    world_height=WORLD_HEIGHT,
                )

                self.data_collector.record_action(
                    movement_dir, state_transition, confidence
                )

                # Execute action
                self._execute_enemy_action(enemy, movement_dir, state_transition)

                # Update enemy
                enemy.update(
                    dt=self.dt,
                    food_positions=[Vector2(food.rect.center) for food in food_group],
                    player_position=Vector2(WORLD_WIDTH / 2, WORLD_HEIGHT / 2),
                    player_size=50.0,
                    other_enemies=[e for e in enemies if e != enemy],
                )

                # Check for food consumption
                food_eaten = 0
                for food in food_group[:]:
                    if enemy.check_collision_with_food(food):
                        old_size = enemy.size
                        if enemy.eat_food(food):
                            food_group.remove(food)
                            food_eaten += 1
                            size_gain = enemy.size - old_size
                            if size_gain > 0:
                                print(
                                    f"-Enemy ate food: {old_size:.1f} → {enemy.size:.1f} (+{size_gain:.1f})"
                                )

                # Check for enemy consumption (Phase 3: Strategic behavior)
                enemies_eaten_this_step = 0
                for other_enemy in enemies[:]:
                    if (
                        other_enemy != enemy
                        and other_enemy.is_active
                        and enemy.check_collision_with_enemy(other_enemy)
                        and enemy.can_eat_enemy(other_enemy)
                    ):

                        old_size = enemy.size
                        if enemy.eat_enemy(other_enemy):
                            enemies_eaten_this_step += 1
                            enemy.enemies_eaten += 1  # Track total enemies eaten
                            size_gain = enemy.size - old_size
                            print(
                                f"👹 Enemy ate another enemy: {old_size:.1f} → {enemy.size:.1f} (+{size_gain:.1f}) - Total enemies eaten: {enemy.enemies_eaten}"
                            )

                # Calculate reward
                reward = self._calculate_step_reward(
                    enemy, food_eaten, enemies_eaten_this_step, episode_time
                )
                self.data_collector.record_reward(reward)

            # Update episode time
            episode_time += self.dt

            # Update active enemies list
            active_enemies = [enemy for enemy in enemies if enemy.is_active]

            # Enhanced early termination for successful episodes
            if episode_time >= min_survival_time:
                for enemy in active_enemies:
                    size_increase = enemy.size - ENEMY_MIN_SIZE

                    # NEW GOAL Success: Reached 300 size quickly
                    if (
                        size_increase >= target_size_increase
                        and episode_time <= max_time_for_target
                    ):
                        print(
                            f"TARGET SUCCESS: AI grew from {ENEMY_MIN_SIZE} to {enemy.size:.1f} "
                            f"(+{size_increase:.1f}) in {episode_time:.1f}s - UNDER 30s TARGET!"
                        )
                        break
                    # Partial Success: Reached target but took too long
                    elif size_increase >= target_size_increase:
                        print(
                            f"TARGET PARTIAL: AI grew from {ENEMY_MIN_SIZE} to {enemy.size:.1f} "
                            f"(+{size_increase:.1f}) in {episode_time:.1f}s - OVER 30s target"
                        )
                        break
                else:
                    continue  # No AI reached target, continue episode
                break  # At least one AI succeeded, end episode

        # End episode - track best performing AI
        if active_enemies:
            # Find the AI with the best performance
            best_enemy = max(active_enemies, key=lambda e: e.size)
            best_size_increase = best_enemy.size - ENEMY_MIN_SIZE
            total_food_eaten = sum(enemy.total_mass_gained for enemy in enemies)

            total_enemies_eaten = sum(enemy.enemies_eaten for enemy in enemies)
            print(
                f"🏆 Best AI: Size {best_enemy.size:.1f} (+{best_size_increase:.1f}), Food eaten: {total_food_eaten}, Enemies eaten: {total_enemies_eaten}"
            )
        else:
            best_enemy = enemies[0]  # Fallback if all died
            best_size_increase = 0
            total_food_eaten = 0

        # Calculate total enemies eaten by all active enemies
        total_enemies_eaten = sum(enemy.enemies_eaten for enemy in enemies)

        self.data_collector.end_episode(
            final_enemy_size=best_enemy.size,
            survival_time=episode_time,
            food_eaten=int(total_food_eaten),
            enemies_eaten=total_enemies_eaten,  # Track actual enemies eaten
            died=len(active_enemies) == 0,
            cause_of_death=(
                "timeout" if episode_time >= self.max_episode_time else "unknown"
            ),
        )

        # Return episode data
        return self.data_collector.episode_data[-1]

    def _get_enemy_state(
        self,
        enemy: EnemyBlob,
        all_enemies: List[EnemyBlob],
        food_group: List[Food],
        game_time: float,
    ) -> torch.Tensor:
        """Get processed state tensor for an enemy"""
        # Get closest food
        food_distances = []
        for food in food_group:
            distance = (
                (enemy.position.x - food.rect.centerx) ** 2
                + (enemy.position.y - food.rect.centery) ** 2
            ) ** 0.5
            food_distances.append((distance, food))

        food_distances.sort(key=lambda x: x[0])
        closest_food = food_distances[:5]

        # Get other enemies
        other_enemies = [e for e in all_enemies if e != enemy and e.is_active]
        enemy_distances = []
        for other_enemy in other_enemies:
            distance = (
                (enemy.position.x - other_enemy.position.x) ** 2
                + (enemy.position.y - other_enemy.position.y) ** 2
            ) ** 0.5
            enemy_distances.append((distance, other_enemy))

        enemy_distances.sort(key=lambda x: x[0])
        closest_enemies = enemy_distances[:3]

        # Process state
        return self.state_processor.process_state(
            enemy_position=(enemy.position.x, enemy.position.y),
            enemy_size=enemy.size,
            enemy_velocity=(enemy.velocity.x, enemy.velocity.y),
            player_position=(WORLD_WIDTH / 2, WORLD_HEIGHT / 2),  # Simulated player
            player_size=50.0,  # Simulated player size
            food_positions=[
                (food.rect.centerx, food.rect.centery) for _, food in closest_food
            ],
            food_sizes=[food.size for _, food in closest_food],
            other_enemy_positions=[
                (e.position.x, e.position.y) for _, e in closest_enemies
            ],
            other_enemy_sizes=[e.size for _, e in closest_enemies],
            game_time=game_time,
        )

    def _execute_enemy_action(
        self, enemy: EnemyBlob, movement_dir: int, state_transition: int
    ):
        """Execute an action for an enemy"""
        # Convert movement direction to velocity
        directions = [
            (0, -1),  # 0: Up
            (1, -1),  # 1: Up-Right
            (2, 0),  # 2: Right
            (1, 1),  # 3: Down-Right
            (0, 1),  # 4: Down
            (-1, 1),  # 5: Down-Left
            (-1, 0),  # 6: Left
            (-1, -1),  # 7: Up-Left
        ]

        if 0 <= movement_dir < len(directions):
            dx, dy = directions[movement_dir]
            enemy.velocity = Vector2(dx * enemy.speed, dy * enemy.speed)

        # Update AI state based on state transition
        states = ["seeking", "hunting", "fleeing", "idle"]
        if 0 <= state_transition < len(states):
            enemy.ai_state = states[state_transition]

    def _calculate_step_reward(
        self, enemy: EnemyBlob, food_eaten: int, enemies_eaten: int, episode_time: float
    ) -> float:
        """Enhanced reward system for beginner AI with 3 phases"""
        base_reward = 0.0
        size_increase = enemy.size - ENEMY_MIN_SIZE

        # NEW GOAL: Aggressive food consumption for 300 size in 30 seconds
        if food_eaten > 0:
            # Balanced food eating reward - primary driver for size growth
            base_reward += food_eaten * 25.0  # Reduced from 50.0 for better learning

            # Bonus for consecutive food eating (encourages sustained eating)
            if hasattr(enemy, "consecutive_food_eaten"):
                enemy.consecutive_food_eaten += food_eaten
            else:
                enemy.consecutive_food_eaten = food_eaten

            # Consecutive eating bonus (exponential reward for sustained eating)
            if enemy.consecutive_food_eaten >= 5:
                consecutive_bonus = enemy.consecutive_food_eaten * 10.0
                base_reward += consecutive_bonus
                print(
                    f"Consecutive eating: {enemy.consecutive_food_eaten} foods! Bonus: +{consecutive_bonus}"
                )
        else:
            # Reset consecutive counter if no food eaten
            if hasattr(enemy, "consecutive_food_eaten"):
                enemy.consecutive_food_eaten = 0

        # Phase 2: Enhanced wall avoidance system to prevent wall-riding
        wall_penalty = self._calculate_wall_avoidance_penalty(enemy)
        base_reward += wall_penalty

        # Phase 3: Strategic behavior - eat enemies (MAIN GOAL for beginner AI)
        if enemies_eaten > 0:
            # High reward for eating enemies - this is the main goal!
            base_reward += (
                enemies_eaten * 200.0
            )  # Reduced from 500.0 for better learning
            print(
                f"Strategic success: Enemy ate {enemies_eaten} enemy(ies)! Reward: +{enemies_eaten * 200.0}"
            )

        # Size growth reward with milestone-based bonuses
        if size_increase > 0:
            # Base growth reward (increased for faster learning)
            base_reward += size_increase * 8.0  # Increased from 3.0 to 8.0

            # Size milestone bonuses (encourages rapid growth) - only trigger once per milestone
            if size_increase >= 150.0 and 150 not in enemy.milestones_achieved:
                base_reward += 100.0  # First milestone
                enemy.milestones_achieved.add(150)
                print(f"Milestone: Reached 150 size increase! Bonus: +100")
            if size_increase >= 200.0 and 200 not in enemy.milestones_achieved:
                base_reward += 200.0  # Second milestone
                enemy.milestones_achieved.add(200)
                print(f"Milestone: Reached 200 size increase! Bonus: +200")
            if size_increase >= 250.0 and 250 not in enemy.milestones_achieved:
                base_reward += 500.0  # Third milestone
                enemy.milestones_achieved.add(250)
                print(f"Milestone: Reached 250 size increase! Bonus: +500")
            if size_increase >= 300.0 and 300 not in enemy.milestones_achieved:
                base_reward += 1000.0  # TARGET MILESTONE!
                enemy.milestones_achieved.add(300)
                print(f"TARGET ACHIEVED: Reached 300 size increase! Bonus: +1000")

            # Time-based urgency bonus (encourages speed)
            if episode_time <= 30.0:
                growth_rate = size_increase / max(episode_time, 1.0)
                base_reward += growth_rate * 200.0  # Increased from 100.0 to 200.0

                # Special bonus for reaching 300 size quickly
                if size_increase >= 300.0:
                    time_bonus = (
                        max(0, 30.0 - episode_time) * 50.0
                    )  # Increased time bonus
                    base_reward += time_bonus
                    print(
                        f"Speed success: Reached 300 size in {episode_time:.1f}s! Time bonus: +{time_bonus}"
                    )

        # Survival reward (decreases over time to encourage action)
        survival_bonus = max(0, 15.0 - episode_time * 0.2)
        base_reward += survival_bonus

        # NEW: Exploration bonus - reward for moving to new areas
        if not hasattr(enemy, "visited_positions"):
            enemy.visited_positions = set()

        # Discretize position for exploration tracking
        grid_x = int(enemy.position.x // 100)
        grid_y = int(enemy.position.y // 100)
        position_key = (grid_x, grid_y)

        if position_key not in enemy.visited_positions:
            enemy.visited_positions.add(position_key)
            base_reward += 1.0  # Small exploration bonus

        # NEW: Center positioning bonus - encourage staying in the middle area
        center_x = WORLD_WIDTH / 2
        center_y = WORLD_HEIGHT / 2
        distance_from_center = (
            (enemy.position.x - center_x) ** 2 + (enemy.position.y - center_y) ** 2
        ) ** 0.5

        # Bonus for being in the center area (within 30% of world size from center)
        max_center_distance = min(WORLD_WIDTH, WORLD_HEIGHT) * 0.3
        if distance_from_center < max_center_distance:
            center_bonus = (
                (max_center_distance - distance_from_center) / max_center_distance * 2.0
            )
            base_reward += center_bonus

        # NEW: Efficiency penalty - discourage excessive movement without eating
        if not hasattr(enemy, "total_movement"):
            enemy.total_movement = 0.0

        # Track movement (simplified - just add a small amount each step)
        enemy.total_movement += 1.0

        # Penalty if moving too much without eating
        if enemy.total_movement > 100 and food_eaten == 0 and enemies_eaten == 0:
            base_reward -= 0.5  # Small penalty for inefficient movement

        return base_reward

    def _calculate_wall_avoidance_penalty(self, enemy: EnemyBlob) -> float:
        """Calculate comprehensive wall avoidance penalty to prevent wall-riding"""
        penalty = 0.0

        # Get distances to all walls
        left_distance = enemy.position.x
        right_distance = WORLD_WIDTH - enemy.position.x
        top_distance = enemy.position.y
        bottom_distance = WORLD_HEIGHT - enemy.position.y

        min_distance = min(left_distance, right_distance, top_distance, bottom_distance)

        # Progressive penalty system - the closer to walls, the higher the penalty
        if min_distance < 20:  # Very close to wall - severe penalty
            penalty -= 15.0
            if not hasattr(enemy, "wall_warning_shown"):
                enemy.wall_warning_shown = False
            if not enemy.wall_warning_shown:
                print(
                    f"WALL WARNING: Enemy too close to wall! Distance: {min_distance:.1f}"
                )
                enemy.wall_warning_shown = True
        elif min_distance < 50:  # Close to wall - moderate penalty
            penalty -= 8.0
        elif min_distance < 100:  # Near wall - small penalty
            penalty -= 3.0
        else:  # Safe distance from walls - small bonus
            penalty += 1.0

        # Corner penalty - being in corners is especially bad
        corner_threshold = 80
        if (
            min_distance < corner_threshold
            and (left_distance < corner_threshold or right_distance < corner_threshold)
            and (top_distance < corner_threshold or bottom_distance < corner_threshold)
        ):
            penalty -= 10.0  # Additional corner penalty
            if not hasattr(enemy, "corner_warning_shown"):
                enemy.corner_warning_shown = False
            if not enemy.corner_warning_shown:
                print(
                    f"CORNER WARNING: Enemy stuck in corner! Distance: {min_distance:.1f}"
                )
                enemy.corner_warning_shown = True

        # Wall-riding detection - if enemy stays near same wall for too long
        if not hasattr(enemy, "wall_history"):
            enemy.wall_history = []

        # Track which wall the enemy is closest to
        if min_distance < 100:
            if left_distance == min_distance:
                wall_type = "left"
            elif right_distance == min_distance:
                wall_type = "right"
            elif top_distance == min_distance:
                wall_type = "top"
            else:
                wall_type = "bottom"

            enemy.wall_history.append(wall_type)

            # Keep only last 20 wall interactions
            if len(enemy.wall_history) > 20:
                enemy.wall_history = enemy.wall_history[-20:]

            # Check for wall-riding (same wall for too long)
            if len(enemy.wall_history) >= 10:
                same_wall_count = sum(
                    1 for w in enemy.wall_history[-10:] if w == wall_type
                )
                if same_wall_count >= 8:  # 80% of last 10 steps on same wall
                    penalty -= 20.0  # Severe wall-riding penalty
                    if not hasattr(enemy, "wall_riding_warning_shown"):
                        enemy.wall_riding_warning_shown = False
                    if not enemy.wall_riding_warning_shown:
                        print(
                            f"WALL-RIDING DETECTED: Enemy riding {wall_type} wall! Penalty: -20.0"
                        )
                        enemy.wall_riding_warning_shown = True
        else:
            # Reset wall history when away from walls
            enemy.wall_history = []

        return penalty

    def train_model(self, episodes: int = None):
        """Train the model for specified number of episodes"""
        if episodes is None:
            episodes = self.training_episodes

        print(f"Starting training for {episodes} episodes...")

        # Start from the highest existing episode + 1
        start_episode = self._get_next_episode_number()

        for episode in tqdm(
            range(start_episode, start_episode + episodes), desc="Training"
        ):
            # Simulate episode
            episode_data = self.simulate_episode()

            # Calculate episode performance
            performance = self._calculate_episode_performance(episode_data)

            # Save model every 5 episodes for progress tracking
            if episode % 5 == 0:
                model_path = (
                    f"ai_training/models/{self.difficulty}_model_episode_{episode}.pth"
                )
                os.makedirs("ai_training/models", exist_ok=True)

                self.model_manager.save_model(
                    self.model,
                    episode,
                    {"performance": performance},
                    self.difficulty,
                )
                print(f"Model saved: {model_path}")

            # Update best performance and save best model
            if performance > self.best_performance:
                self.best_performance = performance
                best_model_path = (
                    f"ai_training/models/best_{self.difficulty}_model_{episode}.pth"
                )

                # Ensure models directory exists
                os.makedirs("ai_training/models", exist_ok=True)

                # Save best model
                self.model_manager.save_model(
                    self.model,
                    episode,
                    {
                        "performance": performance,
                        "best_performance": self.best_performance,
                    },
                    self.difficulty,
                )

                print(
                    f"NEW BEST MODEL: {best_model_path} (Performance: {performance:.2f})"
                )
                self.best_model_path = best_model_path

                # Also save as latest model for easy testing
                latest_model_path = (
                    f"ai_training/models/latest_{self.difficulty}_model.pth"
                )
                # Save latest model directly to the correct path
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "model_config": {
                            "input_size": self.model.input_size,
                            "hidden_sizes": self.model.hidden_sizes,
                            "output_size": self.model.output_size,
                        },
                        "episode": episode,
                        "performance_metrics": {"performance": performance},
                        "difficulty": self.difficulty,
                        "timestamp": datetime.now().isoformat(),
                    },
                    latest_model_path,
                )
                print(f"Latest model saved: {latest_model_path}")

            # Print progress every 5 episodes
            if episode % 5 == 0:
                print(
                    f"Episode {episode}: Performance = {performance:.2f}, Best = {self.best_performance:.2f}"
                )

        print(f"✅ Training completed! Best performance: {self.best_performance:.2f}")

        # Save final model
        final_model_path = (
            f"ai_training/models/{self.difficulty}_model_episode_{episodes-1}.pth"
        )
        self.model_manager.save_model(
            self.model,
            episodes - 1,
            {
                "performance": self.best_performance,
                "final_performance": self.best_performance,
            },
            self.difficulty,
        )

        # Ensure we always have a latest model for easy testing
        self._save_latest_model()

        # Save training history
        self._save_training_history()

        # Create comprehensive training summary
        self._create_training_summary()

        # Run automatic cleanup
        self._run_automatic_cleanup()

    def _calculate_episode_performance(self, episode: TrainingEpisode) -> float:
        """Calculate overall performance score for an episode with beginner AI focus"""
        # Weighted combination of different metrics for beginner AI
        survival_score = episode.survival_time / self.max_episode_time
        growth_score = (
            episode.final_enemy_size / 200.0
        )  # Normalize to reasonable size (200 max)
        food_score = episode.food_eaten / 15.0  # Normalize to reasonable food count

        # Phase 3: Strategic behavior - enemies eaten (HIGH WEIGHT for beginner AI)
        enemies_score = min(episode.enemies_eaten / 2.0, 1.0)  # Cap at 2 enemies eaten

        # Death penalty
        death_penalty = 0.0 if not episode.died else -0.5

        # NEW GOAL bonus: Size growth under 30 seconds
        target_bonus = 0.0
        if episode.final_enemy_size >= 300.0 and episode.survival_time <= 30.0:
            target_bonus = 0.5  # Massive bonus for reaching 300 size in 30s
        elif episode.final_enemy_size >= 200.0 and episode.survival_time <= 30.0:
            target_bonus = 0.3  # Good bonus for reaching 200 size in 30s
        elif episode.final_enemy_size >= 300.0:
            target_bonus = 0.2  # Partial bonus for reaching 300 size (but too slow)

        # NEW: Speed bonus for reaching targets quickly
        speed_bonus = 0.0
        if episode.final_enemy_size >= 200.0:
            time_ratio = episode.survival_time / self.max_episode_time
            speed_bonus = max(0, 0.3 - time_ratio * 0.3)  # Bonus for finishing early

        performance = (
            survival_score * 0.15
            + growth_score * 0.35  # Increased growth emphasis
            + food_score * 0.25  # Maintained food weight
            + enemies_score * 0.25  # Slightly reduced enemy emphasis
            + death_penalty
            + target_bonus  # NEW: Bonus for reaching 300 size goal
            + speed_bonus  # New speed component
        )

        return max(0.0, performance)  # Ensure non-negative

    def _get_next_episode_number(self) -> int:
        """Get the next episode number to start from"""
        import glob

        # Find all existing episode models
        pattern = f"ai_training/models/{self.difficulty}_model_episode_*.pth"
        existing_models = glob.glob(pattern)

        if not existing_models:
            return 0

        # Extract episode numbers and find the highest
        episode_numbers = []
        for model_path in existing_models:
            filename = os.path.basename(model_path)
            try:
                # Extract episode number from filename like "beginner_model_episode_47.pth"
                episode_num = int(filename.split("_")[-1].split(".")[0])
                episode_numbers.append(episode_num)
            except (ValueError, IndexError):
                continue

        if episode_numbers:
            next_episode = max(episode_numbers) + 1
            print(
                f"Found existing models up to episode {max(episode_numbers)}, starting from episode {next_episode}"
            )
            return next_episode
        else:
            return 0

    def _save_training_history(self):
        """Save training history to file"""
        filename = f"{self.difficulty}_training_history.json"
        filepath = os.path.join("ai_training/logs", filename)

        # Create training history data
        training_data = {
            "difficulty": self.difficulty,
            "best_performance": self.best_performance,
            "best_model_path": self.best_model_path,
            "training_episodes": self.training_episodes,
            "num_enemies": self.num_enemies,
            "max_episode_time": self.max_episode_time,
            "training_date": datetime.now().isoformat(),
            "episode_summaries": [],
        }

        # Add episode summaries if available
        if hasattr(self.data_collector, "episode_data"):
            for episode in self.data_collector.episode_data:
                training_data["episode_summaries"].append(
                    {
                        "episode_id": episode.episode_id,
                        "final_size": episode.final_enemy_size,
                        "survival_time": episode.survival_time,
                        "food_eaten": episode.food_eaten,
                        "enemies_eaten": episode.enemies_eaten,
                        "died": episode.died,
                    }
                )

        with open(filepath, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"Training history saved to {filepath}")

    def _save_latest_model(self):
        """Always save the latest model for easy testing"""
        latest_model_path = f"ai_training/models/latest_{self.difficulty}_model.pth"
        os.makedirs("ai_training/models", exist_ok=True)

        # Get the last episode number
        last_episode = self.training_episodes - 1

        self.model_manager.save_model(
            self.model,
            last_episode,
            {"performance": self.best_performance},
            self.difficulty,
        )
        print(f"Latest model saved: {latest_model_path}")

    def _create_training_summary(self):
        """Create a comprehensive training summary report"""
        import time

        # Calculate training statistics
        total_episodes = (
            len(self.data_collector.episode_data)
            if hasattr(self.data_collector, "episode_data")
            else 0
        )
        episodes_with_enemies_eaten = 0
        episodes_reaching_150_size = 0
        episodes_reaching_200_size = 0
        total_food_eaten = 0
        total_enemies_eaten = 0
        avg_survival_time = 0
        avg_final_size = 0

        if (
            hasattr(self.data_collector, "episode_data")
            and self.data_collector.episode_data
        ):
            for episode in self.data_collector.episode_data:
                if episode.enemies_eaten > 0:
                    episodes_with_enemies_eaten += 1
                if episode.final_enemy_size >= 150:
                    episodes_reaching_150_size += 1
                if episode.final_enemy_size >= 200:
                    episodes_reaching_200_size += 1
                total_food_eaten += episode.food_eaten
                total_enemies_eaten += episode.enemies_eaten
                avg_survival_time += episode.survival_time
                avg_final_size += episode.final_enemy_size

            avg_survival_time /= total_episodes
            avg_final_size /= total_episodes

        # Create summary data
        summary_data = {
            "training_session": {
                "difficulty": self.difficulty,
                "start_episode": 28,  # Starting from episode 27
                "total_episodes": total_episodes,
                "training_date": datetime.now().isoformat(),
                "best_performance": self.best_performance,
                "best_model_path": self.best_model_path,
            },
            "performance_metrics": {
                "episodes_with_enemies_eaten": episodes_with_enemies_eaten,
                "enemy_eating_rate": (
                    f"{(episodes_with_enemies_eaten/total_episodes*100):.1f}%"
                    if total_episodes > 0
                    else "0%"
                ),
                "episodes_reaching_150_size": episodes_reaching_150_size,
                "episodes_reaching_200_size": episodes_reaching_200_size,
                "avg_survival_time": f"{avg_survival_time:.1f}s",
                "avg_final_size": f"{avg_final_size:.1f}",
                "total_food_eaten": int(total_food_eaten),
                "total_enemies_eaten": total_enemies_eaten,
            },
            "training_goals": {
                "primary_goal": "Consistently gain 200 size in under 15 seconds",
                "milestone_bonuses": "150, 200, 250, 300 size increases",
                "strategic_behavior": "Eat enemies for size advantage",
                "efficient_movement": "Avoid walls and seek food",
            },
            "model_info": {
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "learning_rate": self.learning_rate,
                "max_episode_time": self.max_episode_time,
                "num_enemies_per_episode": self.num_enemies,
            },
        }

        # Save summary to file
        summary_filename = f"{self.difficulty}_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_filepath = os.path.join("ai_training/logs", summary_filename)

        with open(summary_filepath, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Print summary to console
        print(f"\n{'='*60}")
        print(f"TRAINING SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Difficulty: {self.difficulty}")
        print(f"Episodes Trained: {total_episodes}")
        print(f"Best Performance: {self.best_performance:.2f}")
        print(
            f"Best Model: {os.path.basename(self.best_model_path) if self.best_model_path else 'None'}"
        )
        print(f"\nPerformance Metrics:")
        print(
            f"  Enemy Eating Rate: {(episodes_with_enemies_eaten/total_episodes*100):.1f}% ({episodes_with_enemies_eaten}/{total_episodes})"
        )
        print(f"  Episodes Reaching 150+ Size: {episodes_reaching_150_size}")
        print(f"  Episodes Reaching 200+ Size: {episodes_reaching_200_size}")
        print(f"  Average Survival Time: {avg_survival_time:.1f}s")
        print(f"  Average Final Size: {avg_final_size:.1f}")
        print(f"  Total Food Eaten: {int(total_food_eaten)}")
        print(f"  Total Enemies Eaten: {total_enemies_eaten}")
        print(f"\nSummary saved to: {summary_filepath}")
        print(f"{'='*60}")

    def _run_automatic_cleanup(self):
        """Run automatic cleanup after training"""
        try:
            from ai_training.manage_models import automatic_cleanup_after_training

            automatic_cleanup_after_training()
        except ImportError as e:
            print(f"Warning: Could not run automatic cleanup: {e}")
        except Exception as e:
            print(f"Warning: Error during automatic cleanup: {e}")


def main():
    """Main training function"""
    print("Agar.io AI Training System")
    print("=" * 50)

    # Create trainer for beginner difficulty with optimized settings
    trainer = AITrainer(
        difficulty="beginner",
        num_enemies=25,  # Increased for faster learning
        training_episodes=150,  # Extended training session for comprehensive learning
    )

    print("\nBeginner AI Training Goals (UPDATED):")
    print("   PRIMARY GOAL: Consistently gain 200 size in under 15 seconds")
    print("   - Aggressive food consumption with massive rewards")
    print("   - Efficient movement - avoid walls/corners and seek food")
    print("   - Strategic behavior - eat enemies for size advantage")
    print("   - Size milestone bonuses: 175, 200, 250, 300 size increases")

    # Start training
    trainer.train_model()

    print(f"\nTraining Complete!")
    print(f"   Best Performance: {trainer.best_performance:.2f}")
    if trainer.best_model_path:
        print(f"   Best Model: {trainer.best_model_path}")
        print(f"   Latest Model: ai_training/models/latest_beginner_model.pth")

    # Print final summary
    summary = trainer.data_collector.get_training_summary()
    print("\nFinal Training Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
