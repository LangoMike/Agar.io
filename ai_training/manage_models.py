"""
AI Model Management Utility
Helps clean up training data and manage trained models
"""

import os
import shutil
import glob
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ai_training.neural_network import ModelManager
except ImportError:
    ModelManager = None


def cleanup_training_data():
    """Remove episode folders to save space - keep only the trained models"""
    print("🧹 Cleaning up training data...")

    # Find all episode folders
    episode_folders = glob.glob("ai_training/data/episode_*")

    if not episode_folders:
        print("   No episode folders found to clean up")
        return

    print(f"   Found {len(episode_folders)} episode folders")

    # Calculate total size
    total_size = 0
    for folder in episode_folders:
        total_size += sum(
            f.stat().st_size for f in Path(folder).rglob("*") if f.is_file()
        )

    print(f"   Total size: {total_size / (1024*1024):.1f} MB")

    # Ask for confirmation
    response = input("   Delete these folders to save space? (y/N): ")
    if response.lower() == "y":
        for folder in episode_folders:
            shutil.rmtree(folder)
        print("   Episode folders deleted")
    else:
        print("   ❌ Cleanup cancelled")


def automatic_cleanup_after_training():
    """Automatically clean up after training - no user input required"""
    print("🧹 Automatic cleanup after training...")

    # Clean up episode JSON files from ai_training/data
    cleanup_episode_json_files()

    # Clean up excess models (keep only 9 best)
    cleanup_excess_models()

    print("✅ Automatic cleanup completed")


def cleanup_episode_json_files():
    """Delete all episode JSON files from ai_training/data"""
    data_dir = "ai_training/data"
    if not os.path.exists(data_dir):
        return

    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if json_files:
        print(f"   Deleting {len(json_files)} episode JSON files...")
        for json_file in json_files:
            try:
                os.remove(json_file)
            except Exception as e:
                print(f"   Warning: Could not delete {json_file}: {e}")
        print("   Episode JSON files deleted")
    else:
        print("   No episode JSON files found to delete")


def cleanup_excess_models():
    """Keep only the 9 best models: Best, Latest, and 7 best performing episodes"""
    models_dir = "ai_training/models"
    if not os.path.exists(models_dir):
        return

    # Find all episode models
    episode_models = glob.glob(os.path.join(models_dir, "beginner_model_episode_*.pth"))

    if len(episode_models) <= 7:
        print(f"   Only {len(episode_models)} episode models found, no cleanup needed")
        return

    print(f"   Found {len(episode_models)} episode models, keeping only 7 best...")

    # Get performance data for each model
    model_performances = []

    if ModelManager:
        model_manager = ModelManager()
        for model_path in episode_models:
            try:
                model, metadata = model_manager.load_model(model_path)
                episode = metadata.get("episode", 0)
                performance = metadata.get("performance_metrics", {}).get(
                    "performance", 0
                )
                model_performances.append((model_path, episode, performance))
            except Exception as e:
                print(f"   Warning: Could not load {model_path}: {e}")
                # If we can't load it, assume it's not good and should be deleted
                model_performances.append((model_path, 0, 0))
    else:
        # Fallback: sort by episode number (higher = more recent)
        for model_path in episode_models:
            filename = os.path.basename(model_path)
            try:
                episode = int(filename.split("_")[-1].split(".")[0])
                model_performances.append(
                    (model_path, episode, episode)
                )  # Use episode as performance
            except:
                model_performances.append((model_path, 0, 0))

    # Sort by performance (descending), then by episode (descending) for ties
    model_performances.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Keep the 7 best models
    models_to_keep = model_performances[:7]
    models_to_delete = model_performances[7:]

    print(f"   Keeping {len(models_to_keep)} best models:")
    for model_path, episode, performance in models_to_keep:
        print(f"     Episode {episode}: Performance {performance:.2f}")

    print(f"   Deleting {len(models_to_delete)} excess models:")
    for model_path, episode, performance in models_to_delete:
        try:
            os.remove(model_path)
            print(f"     Deleted Episode {episode}")
        except Exception as e:
            print(f"     Warning: Could not delete {model_path}: {e}")

    print("   Model cleanup completed")


def list_available_models():
    """List all available trained models"""
    print("\nAvailable AI Models:")

    # Find all .pth model files
    model_files = glob.glob("ai_training/models/*.pth")

    if not model_files:
        print("   No trained models found")
        return []

    models = []
    for model_file in sorted(model_files):
        filename = os.path.basename(model_file)
        size = os.path.getsize(model_file) / (1024 * 1024)  # MB

        # Determine model type
        if "best_" in filename:
            model_type = "BEST"
        elif "latest_" in filename:
            model_type = "LATEST"
        else:
            model_type = "Episode"

        print(f"   {model_type}: {filename} ({size:.1f} MB)")
        models.append(model_file)

    return models


def copy_model_for_testing(source_model, target_name="test_model.pth"):
    """Copy a model to a testing location"""
    target_path = f"ai_training/models/{target_name}"

    try:
        shutil.copy2(source_model, target_path)
        print(f"Model copied to {target_path}")
        print(f"   You can now run 'python main.py' to test this AI!")
        return True
    except Exception as e:
        print(f"❌ Failed to copy model: {e}")
        return False


def main():
    """Main management interface"""
    print("Agar.io AI Model Manager")
    print("=" * 40)

    while True:
        print("\nOptions:")
        print("1. Clean up training data (save space)")
        print("2. List available models")
        print("3. Copy model for testing")
        print("4. Exit")

        choice = input("\nChoose an option (1-4): ")

        if choice == "1":
            cleanup_training_data()

        elif choice == "2":
            models = list_available_models()

        elif choice == "3":
            models = list_available_models()
            if models:
                print(f"\nSelect a model to copy (1-{len(models)}):")
                for i, model in enumerate(models, 1):
                    print(f"   {i}. {os.path.basename(model)}")

                try:
                    model_choice = int(input("Model number: ")) - 1
                    if 0 <= model_choice < len(models):
                        copy_model_for_testing(models[model_choice])
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Please enter a number")

        elif choice == "4":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
