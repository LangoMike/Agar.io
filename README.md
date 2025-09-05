# Agar.io - Single Player Edition

A Python-based single-player rendition of the popular web game Agar.io, featuring AI-controlled opponents, advanced split mechanics, and neural network training capabilities.

## Game Overview

Agar.io is a competitive multiplayer game where players control circular blobs that can eat food to grow larger, split into multiple pieces for strategic movement, and consume smaller opponents. This single-player version features intelligent AI adversaries and advanced game mechanics.

### Gameplay Objectives

**Primary Goal**: Grow as large as possible by consuming food and other blobs while avoiding being eaten by larger opponents.

**Victory Conditions**:
- **Eliminate all AI enemies** - Become the last blob standing
- **Reach 20,000 mass** - Achieve massive size dominance
- **Survive as long as possible** - Compete for high scores

**Core Mechanics**:
- **Growth**: Eat food pellets to increase your size and mass
- **Combat**: Consume smaller enemies and avoid larger ones
- **Strategy**: Use splitting to escape danger or hunt smaller prey
- **Survival**: Navigate a world filled with 20 AI opponents

## Features

### Core Gameplay
- **Dynamic Player Movement**: Mouse-based movement with size-based speed scaling
- **Food System**: Varied food sizes (19-55 mass) with probability-based distribution and diminishing returns growth
- **Intelligent Food Consumption**: Only eat food if player is bigger than the food
- **Precise Collision Detection**: Food must be touched at center to be consumed
- **Manual Camera Zoom**: Mouse wheel zoom control (0.10x to 3.0x) for user-defined view adjustment
- **World Boundaries**: Proper world constraints with smooth edge handling
- **Diminishing Returns Growth System**: Logarithmic growth scaling prevents runaway expansion
- **AI Adversaries**: 20 intelligent enemy blobs with state-based behavior and enemy vs enemy combat

### Advanced Split Mechanics
- **Multi-Level Splitting**: Split into 2, 4, 8, or 16 blobs (up to 4 splits)
- **Individual Blob Management**: Each split blob tracked independently with individual mass and speed
- **Main Blob Tracking**: Camera follows the designated "main" blob for consistent control
- **Collision Avoidance**: Split blobs automatically avoid overlapping using repulsive forces
- **Individual Growth**: Each split blob can consume food independently, gaining mass in the specific blob that ate it
- **Individual Death**: Split blobs can die independently - only the eaten blob disappears
- **Speed Scaling**: Each blob moves at speed based on its individual mass (larger = slower)
- **Smooth Rejoin Animation**: 3-second countdown with dual movement (user control + automatic convergence)
- **Mass Conservation**: Total mass preserved during splits and rejoins
- **Split Timer UI**: Minimalist countdown timer showing time until rejoin

### AI Adversary System
- **State-Based Behavior**: Enemies operate with seeking, hunting, fleeing, and idle states
- **Intelligent Decision Making**: AI responds to player size, distance, and food availability
- **Dynamic Spawning**: 20 AI enemies with safe spawn positioning and enemy vs enemy combat
- **Enemy vs Enemy Combat**: AI can eat each other with 1% size advantage requirement
- **Neural Network Training**: PyTorch-based reinforcement learning system for AI improvement
- **Automated Training**: AI vs AI battles for data collection and model training
- **Model Management**: Save/load trained models with performance tracking
- **Difficulty Scaling**: Beginner, Intermediate, and Hard AI models
- **Victory Conditions**: Game ends when all AI enemies are eliminated or player reaches 20,000 mass

### Visual & UI
- **Gradient Background**: Smooth dark-to-light gradient background for better movement visibility
- **Manual Zoom Control**: Mouse wheel zoom for user-controlled view adjustment
- **Minimap**: Bottom-left minimap showing player position and world overview
- **Pulse Animation**: Subtle visual effects on player blobs
- **Modern UI**: Clean, informative display with player stats and game info
- **Leaderboard System**: Top 10 player ranking with individual entry borders and right-aligned scores
- **Toggle Controls**: Press "C" to view/hide game controls in a collapsible panel
- **Rounded UI Elements**: Modern rounded corners on all UI components
- **Split Timer Display**: Contextual countdown timer for split rejoin timing
- **Rejoin Animation Indicator**: Visual feedback during the 3-second rejoin process
- **Smooth Rendering**: Optimized graphics with minimal visual artifacts

### Technical Features
- **Modular Architecture**: Clean separation of concerns with dedicated packages
- **Efficient Collision Detection**: Optimized algorithms for smooth performance
- **Memory Management**: Proper lifecycle management for game entities
- **Error Handling**: Comprehensive error handling with graceful fallbacks
- **Performance Optimization**: Efficient rendering and update loops
- **Growth System**: Logarithmic diminishing returns prevents exponential blob expansion
- **UI State Management**: Toggle-based controls system with collapsible panels
- **AI Framework**: Complete neural network training pipeline with PyTorch
- **GPU Acceleration**: DirectML support for AMD GPUs on Windows
- **Batch Processing**: Optimized AI decision-making for multiple enemies
- **Model Persistence**: Save/load trained AI models with metadata

## Detailed Split Mechanics

### How Splitting Works
1. **Split Trigger**: Press SPACE when your blob is large enough (minimum 40 mass for first split)
2. **Split Process**: Your blob divides into 2 smaller blobs, each with half the original mass
3. **Positioning**: Blobs are positioned on opposite sides to avoid immediate overlap
4. **Individual Control**: All blobs move toward your mouse cursor simultaneously
5. **Independent Growth**: Each blob can eat food independently, growing only that specific blob
6. **Speed Scaling**: Each blob moves at speed based on its individual mass
7. **Collision Avoidance**: Blobs automatically avoid overlapping with each other

### Split Strategy
- **Escape**: Split to escape from larger enemies (smaller blobs move faster)
- **Hunt**: Use multiple blobs to corner and consume smaller enemies
- **Growth**: Split blobs can eat food simultaneously for faster growth
- **Risk Management**: Smaller blobs are more vulnerable to being eaten

### Rejoin Process
1. **Timer**: 30-second countdown begins when you split
2. **Animation**: Last 3 seconds show smooth convergence animation
3. **Dual Movement**: You maintain full control while blobs automatically move together
4. **Final Position**: Rejoin happens at the main blob's current location (no teleportation)
5. **Mass Conservation**: All individual blob masses combine into one large blob

### Split Limitations
- **Maximum Splits**: 4 splits maximum (16 total blobs)
- **Minimum Size**: Each blob must be at least 20 mass to split further
- **Time Limit**: Automatic rejoin after 30 seconds
- **Vulnerability**: Smaller blobs are easier targets for enemies

## Getting Started

### Prerequisites
- Python 3.8+
- Pygame 2.6.0+

### Installation
1. Clone the repository:
```bash
git clone https://github.com/LangoMike/Agar.io
cd Agar.io
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the game:
```bash
python main.py
```

### Controls
- **WASD**: Move player
- **Mouse**: Set movement target
- **Mouse Wheel**: Zoom in/out (0.10x to 3.0x)
- **Space**: Split player (when size allows)
- **C**: Toggle controls display
- **P**: Pause/Unpause
- **F11**: Toggle fullscreen mode
- **ESC**: Quit game


## Configuration

### Game Settings
- **World Size**: 19200x10800 pixels (20% larger than screen)
- **Food Density**: 1 food per 50,000 pixels
- **Player Speed**: Base 400, scaled by size^0.2
- **Split Limits**: Maximum 4 splits (16 total blobs)
- **Rejoin Time**: 30 seconds
- **Zoom Range**: 0.10x to 3.0x (controlled by mouse wheel)
- **Enemy Count**: 20 AI enemies
- **Victory Condition**: 20,000 mass or eliminate all enemies

### Food Distribution
- **Size Range**: 19-55 mass
- **Probability**: Smaller food more common, larger food rarer
- **Spawning**: New food spawns for each consumed piece
- **Growth System**: Logarithmic diminishing returns (30% base, 10% minimum)

### AI Training Settings
- **Training Episodes**: 25 per difficulty level
- **Batch Processing**: GPU-accelerated decision making
- **Reward System**: Food eating (10x), enemy eating (1000x), growth bonuses
- **Model Saving**: Best and latest models saved automatically
- **Performance Tracking**: Episode-based performance metrics

## Gameplay Tips

### For Beginners
1. **Start Small**: Focus on eating food before engaging enemies
2. **Watch Your Size**: Larger blobs move slower but are harder to eat
3. **Use Splitting Wisely**: Split to escape danger, not just for fun
4. **Stay Aware**: Use the minimap to track enemy positions
5. **Zoom Out**: Use mouse wheel to get a better view of the battlefield

### Advanced Strategies
1. **Split Hunting**: Use multiple blobs to corner smaller enemies
2. **Mass Management**: Keep track of individual blob sizes when split
3. **Timing**: Plan your splits around the 30-second rejoin timer
4. **Enemy Behavior**: Learn AI patterns to predict their movements
5. **Growth Optimization**: Use diminishing returns to plan your growth strategy

## 🐛 Known Issues

- Some visual artifacts may occur during rapid zoom changes

## Contributing
All contributions are appreciated (Someone please help me fix zoom bug)

## Acknowledgments

- Inspired by the original Agar.io game
- Built with Pygame for Python game development
- AI training powered by PyTorch
- Used VSCode's Built in LLM to create comprehensive README and commit messages.

## Development Status

- **Core Gameplay**: ✅ Complete
- **Split System**: ✅ Complete (100%)
- **AI Enemies**: ✅ Complete (100%)
- **Neural Network Training**: ✅ Complete (100%)
- **UI/UX**: ✅ Complete (100%)
- **Sound System**: ⏳ Planned
- **Particle Effects**: ⏳ Planned

---

**Current Version**: 1.1.0
**Last Updated**: September 2025  
**Python Version**: 3.8+  
**Pygame Version**: 2.6.0+  
**PyTorch Version**: 2.0.0+
