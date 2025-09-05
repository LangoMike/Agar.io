"""
Main entry point for the Agar.io game
"""

import os
import glob
from game.game_engine import GameEngine


def list_available_models():
    """List all available trained AI models"""
    print("\nAvailable AI Models:")

    # Find all .pth model files
    model_files = glob.glob("ai_training/models/*.pth")

    if not model_files:
        print("   No trained models found")
        print("   Run 'python ai_training/trainer.py' to train some AI first!")
        return None

    # Sort models with BEST and LATEST first, then episodes
    def model_sort_key(filename):
        if "best_" in filename:
            return (0, filename)  # BEST models first
        elif "latest_" in filename:
            return (1, filename)  # LATEST models second
        elif "episode" in filename:
            return (2, filename)  # Episode models last
        else:
            return (3, filename)  # Custom models at end

    sorted_models = sorted(
        model_files, key=lambda x: model_sort_key(os.path.basename(x))
    )

    models = []
    for i, model_file in enumerate(sorted_models, 1):
        filename = os.path.basename(model_file)
        size = os.path.getsize(model_file) / (1024 * 1024)  # MB

        # Determine model type
        if "best_" in filename:
            model_type = "BEST"
        elif "latest_" in filename:
            model_type = "LATEST"
        elif "episode" in filename:
            model_type = "Episode"
        else:
            model_type = "Custom"

        print(f"   {i}. {model_type}: {filename} ({size:.1f} MB)")
        models.append(model_file)

    return models


def select_ai_model():
    """Let user select which AI model to play against"""
    print("\nSelect Your AI Opponent:")
    print("   (Or press Enter to use default AI)")

    models = list_available_models()
    if not models:
        return None

    try:
        choice = input(
            f"\nChoose AI model (1-{len(models)}) or Enter for default: "
        ).strip()

        if not choice:  # User pressed Enter
            print("   Using default AI behavior")
            return None

        model_choice = int(choice) - 1
        if 0 <= model_choice < len(models):
            selected_model = models[model_choice]
            print(f"   Selected: {os.path.basename(selected_model)}")
            return selected_model
        else:
            print("   Invalid choice, using default AI")
            return None

    except ValueError:
        print("   Invalid input, using default AI")
        return None


def main():
    """Main function to start the game"""
    print("🎮 Starting Agar.io - Single Player Edition")
    print("Initializing game systems...")

    # Let user select AI model
    selected_model = select_ai_model()

    try:
        # Create and run the game engine
        game = GameEngine()

        # If a model was selected, tell the game to use it
        if selected_model:
            print(f"Loading trained AI: {os.path.basename(selected_model)}")
            # TODO: Pass the selected model to the game engine
            # This will be implemented when we integrate the neural network with the game

        print("✅ Game engine initialized successfully!")
        print("Starting game loop...")

        # Run the game
        game.run()

    except Exception as e:
        print(f"Error during game execution: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("Game ended successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
