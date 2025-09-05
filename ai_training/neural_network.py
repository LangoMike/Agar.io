"""
Neural Network Architecture for AI Enemy Training
Implements a policy network for decision making in the Agar.io game
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any
import json
import os


class AgarAINetwork(nn.Module):
    """
    Neural network for AI enemy decision making

    Input features:
    - Enemy position (2D)
    - Enemy size (1D)
    - Enemy velocity (2D)
    - Player position (2D)
    - Player size (1D)
    - Food positions (up to 5, 2D each = 10D)
    - Food sizes (up to 5, 1D each = 5D)
    - Other enemy positions (up to 3, 2D each = 6D)
    - Other enemy sizes (up to 3, 1D each = 3D)
    - Game time (1D)
    - World dimensions (2D)

    Total input features: 2 + 1 + 2 + 2 + 1 + 10 + 5 + 6 + 3 + 1 + 2 = 35 features
    """

    def __init__(
        self,
        input_size: int = 35,
        hidden_sizes: List[int] = [128, 64, 32],
        output_size: int = 12,
    ):  # 8 movement directions + 4 state transitions
        super(AgarAINetwork, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build the network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # Small dropout for regularization
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

    def get_action(
        self, state: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[int, int, float]:
        """
        Get action from network output

        Args:
            state: Input state tensor
            temperature: Temperature for action selection (higher = more random)

        Returns:
            movement_direction: 0-7 (8 directions)
            state_transition: 0-3 (4 states)
            confidence: Action confidence score
        """
        with torch.no_grad():
            output = self.forward(state)

            # Split output into movement and state components
            movement_logits = output[:8]  # First 8 outputs for movement
            state_logits = output[8:]  # Last 4 outputs for state transitions

            # Apply temperature scaling
            movement_probs = F.softmax(movement_logits / temperature, dim=0)
            state_probs = F.softmax(state_logits / temperature, dim=0)

            # Sample actions
            movement_direction = torch.multinomial(movement_probs, 1).item()
            state_transition = torch.multinomial(state_probs, 1).item()

            # Calculate confidence (max probability)
            movement_confidence = movement_probs[movement_direction].item()
            state_confidence = state_probs[state_transition].item()
            overall_confidence = (movement_confidence + state_confidence) / 2

            return movement_direction, state_transition, overall_confidence

    def get_action_probabilities(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities for all possible actions"""
        output = self.forward(state)

        movement_logits = output[:8]
        state_logits = output[8:]

        movement_probs = F.softmax(movement_logits, dim=0)
        state_probs = F.softmax(state_logits, dim=0)

        return movement_probs, state_probs


class StateProcessor:
    """Processes game state into neural network input format"""

    def __init__(self, world_width: float = 19200, world_height: float = 10800):
        self.world_width = world_width
        self.world_height = world_height

    def normalize_position(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize position to [0, 1] range"""
        x, y = position
        return (x / self.world_width, y / self.world_height)

    def normalize_size(self, size: float) -> float:
        """Normalize size to [0, 1] range (assuming max size of 1000)"""
        return min(size / 1000.0, 1.0)

    def normalize_velocity(self, velocity: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize velocity to [-1, 1] range"""
        vx, vy = velocity
        max_speed = 500.0  # Maximum expected speed
        return (vx / max_speed, vy / max_speed)

    def process_state(
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
    ) -> torch.Tensor:
        """Convert game state to neural network input tensor"""

        # Normalize enemy data
        enemy_pos_norm = self.normalize_position(enemy_position)
        enemy_size_norm = self.normalize_size(enemy_size)
        enemy_vel_norm = self.normalize_velocity(enemy_velocity)

        # Normalize player data
        player_pos_norm = self.normalize_position(player_position)
        player_size_norm = self.normalize_size(player_size)

        # Normalize food data (pad to 5 items)
        food_positions_norm = []
        food_sizes_norm = []
        for i in range(5):
            if i < len(food_positions):
                food_positions_norm.extend(self.normalize_position(food_positions[i]))
                food_sizes_norm.append(self.normalize_size(food_sizes[i]))
            else:
                food_positions_norm.extend([0.0, 0.0])  # Padding
                food_sizes_norm.append(0.0)

        # Normalize other enemy data (pad to 3 items)
        other_enemy_positions_norm = []
        other_enemy_sizes_norm = []
        for i in range(3):
            if i < len(other_enemy_positions):
                other_enemy_positions_norm.extend(
                    self.normalize_position(other_enemy_positions[i])
                )
                other_enemy_sizes_norm.append(self.normalize_size(other_enemy_sizes[i]))
            else:
                other_enemy_positions_norm.extend([0.0, 0.0])  # Padding
                other_enemy_sizes_norm.append(0.0)

        # Normalize game time (assuming max game time of 300 seconds)
        game_time_norm = min(game_time / 300.0, 1.0)

        # Combine all features
        features = [
            # Enemy data (5 features)
            *enemy_pos_norm,  # 2
            enemy_size_norm,  # 1
            *enemy_vel_norm,  # 2
            # Player data (3 features)
            *player_pos_norm,  # 2
            player_size_norm,  # 1
            # Food data (15 features)
            *food_positions_norm,  # 10
            *food_sizes_norm,  # 5
            # Other enemies data (9 features)
            *other_enemy_positions_norm,  # 6
            *other_enemy_sizes_norm,  # 3
            # Game context (3 features)
            game_time_norm,  # 1
            1.0,
            1.0,  # 2 (world dimensions, always 1.0 after normalization)
        ]

        return torch.tensor(features, dtype=torch.float32)


class ModelManager:
    """Manages saving and loading of trained models"""

    def __init__(self, models_dir: str = "ai_training/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(
        self,
        model: AgarAINetwork,
        episode: int,
        performance_metrics: Dict[str, float],
        difficulty: str = "beginner",
    ):
        """Save a trained model with metadata"""
        filename = f"{difficulty}_model_episode_{episode}.pth"
        filepath = os.path.join(self.models_dir, filename)

        # Save model state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_size": model.input_size,
                    "hidden_sizes": model.hidden_sizes,
                    "output_size": model.output_size,
                },
                "episode": episode,
                "performance_metrics": performance_metrics,
                "difficulty": difficulty,
            },
            filepath,
        )

        print(f"Model saved: {filename}")
        return filepath

    def load_model(self, filepath: str) -> Tuple[AgarAINetwork, Dict[str, Any]]:
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        # Recreate model
        config = checkpoint["model_config"]
        model = AgarAINetwork(
            input_size=config["input_size"],
            hidden_sizes=config["hidden_sizes"],
            output_size=config["output_size"],
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        metadata = {
            "episode": checkpoint["episode"],
            "performance_metrics": checkpoint["performance_metrics"],
            "difficulty": checkpoint["difficulty"],
        }

        print(f"📂 Model loaded: {filepath}")
        return model, metadata

    def get_best_model(self, difficulty: str = "beginner") -> str:
        """Get the path to the best model for a given difficulty"""
        model_files = [
            f
            for f in os.listdir(self.models_dir)
            if f.startswith(f"{difficulty}_model_")
        ]

        if not model_files:
            return None

        # For now, return the most recent model
        # In a real implementation, you'd compare performance metrics
        model_files.sort(reverse=True)
        return os.path.join(self.models_dir, model_files[0])


def create_beginner_model() -> AgarAINetwork:
    """Create a model specifically for beginner difficulty"""
    return AgarAINetwork(
        input_size=35,
        hidden_sizes=[64, 32],  # Smaller network for simpler behavior
        output_size=12,
    )


def create_intermediate_model() -> AgarAINetwork:
    """Create a model for intermediate difficulty"""
    return AgarAINetwork(
        input_size=35, hidden_sizes=[128, 64, 32], output_size=12  # Standard network
    )


def create_hard_model() -> AgarAINetwork:
    """Create a model for hard difficulty"""
    return AgarAINetwork(
        input_size=35,
        hidden_sizes=[256, 128, 64, 32],  # Larger network for complex behavior
        output_size=12,
    )


if __name__ == "__main__":
    # Test the neural network
    print("🧠 Testing Neural Network Architecture...")

    # Create a test model
    model = AgarAINetwork()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test state processor
    processor = StateProcessor()

    # Create a test state
    test_state = processor.process_state(
        enemy_position=(1000.0, 1000.0),
        enemy_size=50.0,
        enemy_velocity=(10.0, 5.0),
        player_position=(2000.0, 2000.0),
        player_size=75.0,
        food_positions=[(1500.0, 1500.0), (1600.0, 1600.0)],
        food_sizes=[19.0, 25.0],
        other_enemy_positions=[(3000.0, 3000.0)],
        other_enemy_sizes=[60.0],
        game_time=10.0,
    )

    print(f"Processed state shape: {test_state.shape}")

    # Test model forward pass
    with torch.no_grad():
        output = model(test_state.unsqueeze(0))
        print(f"Model output shape: {output.shape}")

        # Test action selection
        movement, state, confidence = model.get_action(test_state)
        print(
            f"Selected action - Movement: {movement}, State: {state}, Confidence: {confidence:.3f}"
        )

    # Test model manager
    manager = ModelManager()
    print("All components working correctly!")
