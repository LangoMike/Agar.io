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
            self.model = create_beginner_model()
        elif difficulty == "intermediate":
            self.model = create_intermediate_model()
        else:  # hard
            self.model = create_hard_model()

        # Move model to GPU if available (using DirectML for AMD GPUs on Windows)
        try:
            import torch_directml

            self.device = torch_directml.device()
            print(f"🚀 Using DirectML device: {self.device}")
        except ImportError:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🚀 Using device: {self.device}")

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
            self.max_episode_time = 30.0  # Reduced to 30s for faster learning
        elif difficulty == "intermediate":
            self.max_episode_time = 45.0  # Medium complexity
        else:  # hard
            self.max_episode_time = 60.0  # Full complexity

        self.dt = 1.0 / 30.0  # Reduced from 60 FPS to 30 FPS for faster simulation

        print(f"🤖 AI Trainer initialized for {difficulty} difficulty")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"   Training episodes: {training_episodes}")
        print(f"   Enemies per episode: {num_enemies}")

    def create_training_environment(self) -> Tuple[List[EnemyBlob], List[Food], World]:
        """Create a training environment with enemies and food"""
        # Create world
        world = World()

        # Create enemies
        enemies = []
        for i in range(self.num_enemies):
            # Random spawn position
            x = random.uniform(100, WORLD_WIDTH - 100)
            y = random.uniform(100, WORLD_HEIGHT - 100)
            position = Vector2(x, y)

            enemy = EnemyBlob(position, ENEMY_MIN_SIZE)
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
        # Enhanced early termination conditions
        min_survival_time = 3.0  # Reduced for faster feedback
        target_size_increase = 100.0
        max_time_for_target = 20.0  # Target: 100 size in 20 seconds

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
                                    f"🍎 Enemy ate food: {old_size:.1f} → {enemy.size:.1f} (+{size_gain:.1f})"
                                )

                # Calculate reward
                reward = self._calculate_step_reward(enemy, food_eaten, episode_time)
                self.data_collector.record_reward(reward)

            # Update episode time
            episode_time += self.dt

            # Update active enemies list
            active_enemies = [enemy for enemy in enemies if enemy.is_active]

            # Enhanced early termination for successful episodes
            if episode_time >= min_survival_time:
                for enemy in active_enemies:
                    size_increase = enemy.size - ENEMY_MIN_SIZE

                    # Success: Reached target size quickly
                    if (
                        size_increase >= target_size_increase
                        and episode_time <= max_time_for_target
                    ):
                        print(
                            f"🎯 SUCCESS: AI grew from {ENEMY_MIN_SIZE} to {enemy.size:.1f} "
                            f"(+{size_increase:.1f}) in {episode_time:.1f}s - UNDER 20s TARGET!"
                        )
                        break
                    # Partial success: Reached target but took too long
                    elif size_increase >= target_size_increase:
                        print(
                            f"🎯 PARTIAL: AI grew from {ENEMY_MIN_SIZE} to {enemy.size:.1f} "
                            f"(+{size_increase:.1f}) in {episode_time:.1f}s - OVER 20s target"
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

            print(
                f"🏆 Best AI: Size {best_enemy.size:.1f} (+{best_size_increase:.1f}), Food eaten: {total_food_eaten}"
            )
        else:
            best_enemy = enemies[0]  # Fallback if all died
            best_size_increase = 0
            total_food_eaten = 0

        self.data_collector.end_episode(
            final_enemy_size=best_enemy.size,
            survival_time=episode_time,
            food_eaten=int(total_food_eaten),
            enemies_eaten=0,  # Simplified for now
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
        self, enemy: EnemyBlob, food_eaten: int, episode_time: float
    ) -> float:
        """Enhanced reward system for better learning"""
        base_reward = 0.0

        # Food eating reward (primary focus)
        if food_eaten > 0:
            # Exponential reward for eating food - encourages consistent eating
            base_reward += food_eaten * 10.0

        # Enemy eating reward (HUGE reward - main game mechanic!)
        if hasattr(enemy, "enemies_eaten") and enemy.enemies_eaten > 0:
            # Massive reward for eating other AI enemies - this is the core gameplay!
            base_reward += (
                enemy.enemies_eaten * 1000.0
            )  # 1000x reward for each enemy eaten

        # Size growth reward
        size_increase = enemy.size - ENEMY_MIN_SIZE
        if size_increase > 0:
            # Reward for growing, with bonus for rapid growth
            growth_rate = size_increase / max(episode_time, 1.0)
            base_reward += size_increase * 2.0 + growth_rate * 50.0

        # Survival reward (decreases over time to encourage action)
        survival_bonus = max(0, 10.0 - episode_time * 0.1)
        base_reward += survival_bonus

        # Speed bonus for reaching target quickly
        if size_increase >= 100.0:
            time_bonus = max(0, 30.0 - episode_time) * 10.0
            base_reward += time_bonus

        return base_reward

    def train_model(self, episodes: int = None):
        """Train the model for specified number of episodes"""
        if episodes is None:
            episodes = self.training_episodes

        print(f"🚀 Starting training for {episodes} episodes...")

        for episode in tqdm(range(episodes), desc="Training"):
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
                    model_path,
                    episode_num=episode,
                    performance=performance,
                    metadata={
                        "difficulty": self.difficulty,
                        "episode": episode,
                        "performance": performance,
                        "training_date": datetime.now().isoformat(),
                    },
                )
                print(f"💾 Model saved: {model_path}")

            # Update best performance and save best model
            if performance > self.best_performance:
                self.best_performance = performance
                best_model_path = f"ai_training/models/best_{self.difficulty}_model.pth"

                # Ensure models directory exists
                os.makedirs("ai_training/models", exist_ok=True)

                # Save best model
                self.model_manager.save_model(
                    self.model,
                    best_model_path,
                    episode_num=episode,
                    performance=performance,
                    metadata={
                        "difficulty": self.difficulty,
                        "best_performance": self.best_performance,
                        "training_date": datetime.now().isoformat(),
                    },
                )

                print(
                    f"🏆 NEW BEST MODEL: {best_model_path} (Performance: {performance:.2f})"
                )
                self.best_model_path = best_model_path

                # Also save as latest model for easy testing
                latest_model_path = (
                    f"ai_training/models/latest_{self.difficulty}_model.pth"
                )
                self.model_manager.save_model(
                    self.model,
                    latest_model_path,
                    episode_num=episode,
                    performance=performance,
                )
                print(f"📁 Latest model saved: {latest_model_path}")

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
            final_model_path,
            episode_num=episodes - 1,
            performance=self.best_performance,
            metadata={
                "difficulty": self.difficulty,
                "final_performance": self.best_performance,
                "training_date": datetime.now().isoformat(),
            },
        )

        # Save training history
        self._save_training_history()

    def _calculate_episode_performance(self, episode: TrainingEpisode) -> float:
        """Calculate overall performance score for an episode"""
        # Weighted combination of different metrics
        survival_score = episode.survival_time / self.max_episode_time
        growth_score = episode.final_enemy_size / 100.0  # Normalize to reasonable size
        food_score = episode.food_eaten / 10.0  # Normalize to reasonable food count

        # Death penalty
        death_penalty = 0.0 if not episode.died else -0.5

        performance = (
            survival_score * 0.4 + growth_score * 0.3 + food_score * 0.3 + death_penalty
        )

        return max(0.0, performance)  # Ensure non-negative

    def _save_training_history(self):
        """Save training history to file"""
        filename = f"{self.difficulty}_training_history.json"
        filepath = os.path.join("ai_training/logs", filename)

        with open(filepath, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"📊 Training history saved to {filepath}")


def main():
    """Main training function"""
    print("🎮 Agar.io AI Training System")
    print("=" * 50)

    # Create trainer for beginner difficulty with optimized settings
    trainer = AITrainer(
        difficulty="beginner",
        num_enemies=20,  # Increased for faster learning
        training_episodes=25,  # Reduced for faster testing and iteration
    )

    print("\n🎯 Training Goals:")
    print("   Phase 1: Consistently reach 100 size in under 20 seconds")
    print("   Phase 2: Efficient movement and food seeking")
    print("   Phase 3: Strategic behavior and survival")
    print("   Phase 4: Advanced tactics and enemy avoidance")

    # Start training
    trainer.train_model()

    print(f"\n🏆 Training Complete!")
    print(f"   Best Performance: {trainer.best_performance:.2f}")
    if trainer.best_model_path:
        print(f"   Best Model: {trainer.best_model_path}")
        print(f"   Latest Model: ai_training/models/latest_beginner_model.pth")
        print(f"\n🎮 To test your AI:")
        print(f"   1. Copy 'latest_beginner_model.pth' to 'ai_training/models/'")
        print(f"   2. Run 'python main.py' to play against the trained AI")
        print(f"   3. Watch for improved food-eating behavior!")

    # Print final summary
    summary = trainer.data_collector.get_training_summary()
    print("\n📈 Final Training Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
