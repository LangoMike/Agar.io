"""
AI Model Management Utility
Helps clean up training data and manage trained models
"""

import os
import shutil
import glob
from pathlib import Path


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
        print("   ✅ Episode folders deleted")
    else:
        print("   ❌ Cleanup cancelled")


def list_available_models():
    """List all available trained models"""
    print("\n🤖 Available AI Models:")

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
            model_type = "🏆 BEST"
        elif "latest_" in filename:
            model_type = "📁 LATEST"
        else:
            model_type = "📊 Episode"

        print(f"   {model_type}: {filename} ({size:.1f} MB)")
        models.append(model_file)

    return models


def copy_model_for_testing(source_model, target_name="test_model.pth"):
    """Copy a model to a testing location"""
    target_path = f"ai_training/models/{target_name}"

    try:
        shutil.copy2(source_model, target_path)
        print(f"✅ Model copied to {target_path}")
        print(f"   You can now run 'python main.py' to test this AI!")
        return True
    except Exception as e:
        print(f"❌ Failed to copy model: {e}")
        return False


def main():
    """Main management interface"""
    print("🎮 Agar.io AI Model Manager")
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
