"""
Data Collection System for AI Training
Collects game state data during AI vs AI battles for neural network training
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class GameState:
    """Represents a single game state snapshot"""

    # Enemy information
    enemy_position: Tuple[float, float]
    enemy_size: float
    enemy_velocity: Tuple[float, float]

    # Player information (for context)
    player_position: Tuple[float, float]
    player_size: float

    # Food information (closest 5 food items)
    food_positions: List[Tuple[float, float]]
    food_sizes: List[float]

    # Other enemies (closest 3 enemies)
    other_enemy_positions: List[Tuple[float, float]]
    other_enemy_sizes: List[float]

    # Game context
    game_time: float
    world_width: float
    world_height: float


@dataclass
class Action:
    """Represents an action taken by the AI"""

    # Movement action (8 directions: up, up-right, right, down-right, down, down-left, left, up-left)
    movement_direction: int  # 0-7

    # State transition (seeking, hunting, fleeing, idle)
    state_transition: int  # 0-3

    # Action confidence (0.0 to 1.0)
    confidence: float


@dataclass
class TrainingEpisode:
    """Represents a complete training episode"""

    episode_id: str
    start_time: datetime
    end_time: datetime
    duration: float

    # Episode data
    states: List[GameState]
    actions: List[Action]
    rewards: List[float]

    # Episode outcome
    final_enemy_size: float
    survival_time: float
    food_eaten: int
    enemies_eaten: int
    died: bool
    cause_of_death: str  # "player", "enemy", "timeout"


class DataCollector:
    """Collects and manages training data for AI learning"""

    def __init__(self, data_dir: str = "ai_training/data"):
        self.data_dir = data_dir
        self.current_episode = None
        self.episode_data = []

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Training statistics
        self.total_episodes = 0
        self.total_states = 0
        self.total_actions = 0

    def start_episode(self, episode_id: str = None) -> str:
        """Start a new training episode"""
        if episode_id is None:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_episode = {
            "episode_id": episode_id,
            "start_time": datetime.now(),
            "states": [],
            "actions": [],
            "rewards": [],
            "final_enemy_size": 0,
            "survival_time": 0,
            "food_eaten": 0,
            "enemies_eaten": 0,
            "died": False,
            "cause_of_death": "unknown",
        }

        print(f"Started episode: {episode_id}")
        return episode_id

    def record_state(
        self,
        enemy_position: Tuple[float, float],
        enemy_size: float,
        enemy_velocity: Tuple[float, float],
        player_position: Tuple[float, float],
        player_size: float,
        food_positions: List[Tuple[float, float]],
        food_sizes: List[float],
        other_enemy_positions: List[Tuple[float, float]],
        other_enemy_sizes: List[float],
        game_time: float,
        world_width: float,
        world_height: float,
    ):
        """Record a game state snapshot"""
        if self.current_episode is None:
            raise ValueError("No active episode. Call start_episode() first.")

        # Limit food and enemy data to closest items
        food_data = list(zip(food_positions, food_sizes))[:5]  # Top 5 closest
        enemy_data = list(zip(other_enemy_positions, other_enemy_sizes))[
            :3
        ]  # Top 3 closest

        state = GameState(
            enemy_position=enemy_position,
            enemy_size=enemy_size,
            enemy_velocity=enemy_velocity,
            player_position=player_position,
            player_size=player_size,
            food_positions=[pos for pos, _ in food_data],
            food_sizes=[size for _, size in food_data],
            other_enemy_positions=[pos for pos, _ in enemy_data],
            other_enemy_sizes=[size for _, size in enemy_data],
            game_time=game_time,
            world_width=world_width,
            world_height=world_height,
        )

        self.current_episode["states"].append(state)
        self.total_states += 1

    def record_action(
        self, movement_direction: int, state_transition: int, confidence: float = 1.0
    ):
        """Record an action taken by the AI"""
        if self.current_episode is None:
            raise ValueError("No active episode. Call start_episode() first.")

        action = Action(
            movement_direction=movement_direction,
            state_transition=state_transition,
            confidence=confidence,
        )

        self.current_episode["actions"].append(action)
        self.total_actions += 1

    def record_reward(self, reward: float):
        """Record a reward for the current state-action pair"""
        if self.current_episode is None:
            raise ValueError("No active episode. Call start_episode() first.")

        self.current_episode["rewards"].append(reward)

    def end_episode(
        self,
        final_enemy_size: float,
        survival_time: float,
        food_eaten: int,
        enemies_eaten: int,
        died: bool,
        cause_of_death: str = "unknown",
    ):
        """End the current episode and save data"""
        if self.current_episode is None:
            raise ValueError("No active episode to end.")

        # Update episode data
        self.current_episode["end_time"] = datetime.now()
        self.current_episode["duration"] = (
            self.current_episode["end_time"] - self.current_episode["start_time"]
        ).total_seconds()
        self.current_episode["final_enemy_size"] = final_enemy_size
        self.current_episode["survival_time"] = survival_time
        self.current_episode["food_eaten"] = food_eaten
        self.current_episode["enemies_eaten"] = enemies_eaten
        self.current_episode["died"] = died
        self.current_episode["cause_of_death"] = cause_of_death

        # Create episode object
        episode = TrainingEpisode(
            episode_id=self.current_episode["episode_id"],
            start_time=self.current_episode["start_time"],
            end_time=self.current_episode["end_time"],
            duration=self.current_episode["duration"],
            states=self.current_episode["states"],
            actions=self.current_episode["actions"],
            rewards=self.current_episode["rewards"],
            final_enemy_size=final_enemy_size,
            survival_time=survival_time,
            food_eaten=food_eaten,
            enemies_eaten=enemies_eaten,
            died=died,
            cause_of_death=cause_of_death,
        )

        # Save episode data
        self._save_episode(episode)

        # Update statistics
        self.total_episodes += 1
        self.episode_data.append(episode)

        print(f"Episode {episode.episode_id} completed:")
        print(f"   Duration: {episode.duration:.1f}s")
        print(f"   Final size: {episode.final_enemy_size:.1f}")
        print(f"   Food eaten: {episode.food_eaten}")
        print(f"   Enemies eaten: {episode.enemies_eaten}")
        print(f"   Died: {episode.died} ({episode.cause_of_death})")

        # Reset for next episode
        self.current_episode = None

    def _save_episode(self, episode: TrainingEpisode):
        """Save episode data to file"""
        filename = f"{episode.episode_id}.json"
        filepath = os.path.join(self.data_dir, filename)

        # Convert to serializable format
        episode_dict = asdict(episode)

        # Convert datetime objects to strings
        episode_dict["start_time"] = episode.start_time.isoformat()
        episode_dict["end_time"] = episode.end_time.isoformat()

        # Convert states and actions
        episode_dict["states"] = [asdict(state) for state in episode.states]
        episode_dict["actions"] = [asdict(action) for action in episode.actions]

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Apply conversion to all values
        def convert_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    convert_dict(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            convert_dict(item)
                        else:
                            value[i] = convert_numpy_types(item)
                else:
                    d[key] = convert_numpy_types(value)

        convert_dict(episode_dict)

        with open(filepath, "w") as f:
            json.dump(episode_dict, f, indent=2)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected training data"""
        if not self.episode_data:
            return {"message": "No training data collected yet"}

        total_duration = sum(ep.duration for ep in self.episode_data)
        avg_survival_time = sum(ep.survival_time for ep in self.episode_data) / len(
            self.episode_data
        )
        avg_final_size = sum(ep.final_enemy_size for ep in self.episode_data) / len(
            self.episode_data
        )
        death_rate = sum(1 for ep in self.episode_data if ep.died) / len(
            self.episode_data
        )

        return {
            "total_episodes": self.total_episodes,
            "total_states": self.total_states,
            "total_actions": self.total_actions,
            "total_training_time": total_duration,
            "average_survival_time": avg_survival_time,
            "average_final_size": avg_final_size,
            "death_rate": death_rate,
            "data_directory": self.data_dir,
        }

    def load_episode(self, episode_id: str) -> TrainingEpisode:
        """Load a specific episode from file"""
        filename = f"{episode_id}.json"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Episode {episode_id} not found")

        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct episode object
        episode = TrainingEpisode(
            episode_id=data["episode_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            duration=data["duration"],
            states=[GameState(**state) for state in data["states"]],
            actions=[Action(**action) for action in data["actions"]],
            rewards=data["rewards"],
            final_enemy_size=data["final_enemy_size"],
            survival_time=data["survival_time"],
            food_eaten=data["food_eaten"],
            enemies_eaten=data["enemies_eaten"],
            died=data["died"],
            cause_of_death=data["cause_of_death"],
        )

        return episode


def calculate_reward(
    enemy_size: float,
    food_eaten: int,
    enemies_eaten: int,
    survival_time: float,
    died: bool,
    cause_of_death: str,
) -> float:
    """Calculate reward for an AI action"""
    reward = 0.0

    # Base survival reward
    reward += survival_time * 0.1

    # Growth reward
    reward += enemy_size * 0.01

    # Food consumption reward
    reward += food_eaten * 10.0

    # Enemy consumption reward
    reward += enemies_eaten * 50.0

    # Death penalty
    if died:
        if cause_of_death == "player":
            reward -= 100.0  # Being eaten by player is bad
        elif cause_of_death == "enemy":
            reward -= 50.0  # Being eaten by enemy is also bad
        else:
            reward -= 25.0  # Other death causes

    return reward


if __name__ == "__main__":
    # Test the data collector
    collector = DataCollector()

    # Simulate a training episode
    episode_id = collector.start_episode()

    # Record some sample states and actions
    for i in range(10):
        collector.record_state(
            enemy_position=(100.0 + i, 100.0 + i),
            enemy_size=20.0 + i,
            enemy_velocity=(1.0, 1.0),
            player_position=(200.0, 200.0),
            player_size=25.0,
            food_positions=[(150.0, 150.0), (160.0, 160.0)],
            food_sizes=[19.0, 25.0],
            other_enemy_positions=[(300.0, 300.0)],
            other_enemy_sizes=[30.0],
            game_time=i * 0.1,
            world_width=19200,
            world_height=10800,
        )

        collector.record_action(
            movement_direction=i % 8, state_transition=i % 4, confidence=0.8
        )

        collector.record_reward(1.0 + i * 0.1)

    # End the episode
    collector.end_episode(
        final_enemy_size=30.0,
        survival_time=1.0,
        food_eaten=2,
        enemies_eaten=0,
        died=False,
        cause_of_death="none",
    )

    # Print summary
    summary = collector.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
