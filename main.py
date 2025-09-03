"""
Main entry point for the Agar.io game
"""

from game.game_engine import GameEngine

def main():
    """Main function to start the game"""
    print("🎮 Starting Agar.io - Single Player Edition")
    print("🚀 Initializing game systems...")
    
    try:
        # Create and run the game engine
        game = GameEngine()
        print("✅ Game engine initialized successfully!")
        print("🎯 Starting game loop...")
        
        # Run the game
        game.run()
        
    except Exception as e:
        print(f"❌ Error during game execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("👋 Game ended successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
