# Agar.io - Single Player Edition

A Python-based single-player rendition of the popular web game Agar.io, featuring AI-controlled opponents and advanced game mechanics.

## 🎮 Features

### Core Gameplay
- **Dynamic Player Movement**: Smooth WASD and mouse-based movement with size-based speed scaling
- **Food System**: Varied food sizes (19-55 mass) with probability-based distribution
- **Intelligent Food Consumption**: Only eat food if player is bigger than the food
- **Precise Collision Detection**: Food must be touched at center to be consumed
- **Dynamic Camera System**: Smooth camera following with intelligent zoom based on player size
- **World Boundaries**: Proper world constraints with smooth edge handling

### Split Functionality
- **Multi-Level Splitting**: Split into 2, 4, 8, or 16 blobs (up to 4 splits)
- **Collision Avoidance**: Split blobs automatically avoid overlapping using repulsive forces
- **Intelligent Positioning**: Blobs positioned strategically around the main blob
- **Individual Growth**: Each split blob can consume food independently
- **Automatic Rejoining**: Blobs rejoin after 30 seconds into a single blob
- **Mass Conservation**: Total mass preserved during splits and rejoins

### Visual & UI
- **Petri Dish Background**: Authentic grid pattern resembling the original game
- **Dynamic Zoom**: Camera zooms out as player grows, maintaining optimal view
- **Minimap**: Bottom-left minimap showing player position and world overview
- **Pulse Animation**: Subtle visual effects on player blobs
- **Modern UI**: Clean, informative display with player stats and game info
- **Smooth Rendering**: Optimized graphics with minimal visual artifacts

### Technical Features
- **Modular Architecture**: Clean separation of concerns with dedicated packages
- **Efficient Collision Detection**: Optimized algorithms for smooth performance
- **Memory Management**: Proper lifecycle management for game entities
- **Error Handling**: Comprehensive error handling with graceful fallbacks
- **Performance Optimization**: Efficient rendering and update loops

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Pygame 2.6.0+

### Installation
1. Clone the repository:
```bash
git clone https://github.com/LangoMike/Agar.io
cd Agar.io
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the game:
```bash
python main.py
```

### Controls
- **WASD**: Move player
- **Mouse**: Set movement target
- **Space**: Split player (when size allows)
- **P**: Pause/Unpause
- **ESC**: Quit game

## 🏗️ Project Structure

```
Agar.io/
├── main.py                 # Main entry point
├── game/                   # Core game systems
│   ├── game_engine.py     # Main game loop and coordination
│   ├── camera.py          # Camera and viewport management
│   ├── world.py           # World rendering and boundaries
│   └── ui_manager.py      # User interface and minimap
├── entities/               # Game objects
│   ├── player.py          # Main player and split management
│   ├── split_blob.py      # Individual split blob logic
│   ├── food.py            # Food objects and spawning
│   └── enemy.py           # Enemy blob definitions
├── ai/                     # AI systems (future)
│   ├── enemy_ai.py        # Enemy behavior logic
│   ├── neural_network.py  # TensorFlow AI models
│   └── difficulty.py      # Difficulty level management
├── mechanics/              # Game mechanics
│   ├── collision.py       # Collision detection system
│   ├── split_manager.py   # Split/rejoin logic
│   └── powerups.py        # Power-up system (future)
├── utils/                  # Utility functions
│   ├── constants.py       # Game configuration
│   ├── math_utils.py      # Mathematical utilities
│   └── helpers.py         # Helper functions
├── assets/                 # Game resources
│   ├── sounds/            # Audio files
│   ├── fonts/             # Font files
│   └── particles/         # Particle effects
└── config/                 # Configuration files
    ├── settings.py        # Game settings
    └── difficulty_config.py # Difficulty configurations
```

## 🔧 Configuration

### Game Settings
- **World Size**: 19200x10800 pixels (20% larger than screen)
- **Food Density**: 1 food per 50,000 pixels
- **Player Speed**: Base 400, scaled by size^0.2
- **Split Limits**: Maximum 4 splits (16 total blobs)
- **Rejoin Time**: 30 seconds

### Food Distribution
- **Size Range**: 19-55 mass
- **Probability**: Smaller food more common, larger food rarer
- **Spawning**: New food spawns for each consumed piece

## 🎯 Next Steps

### High Priority
1. **Fix Split Mechanics**
   - Food consumption and growth logic for split blobs
   - Camera control during splits
   - Zoom logic optimization for split states

2. **Fix Background Grid Scaling**
   - Eliminate grid glitches during splits
   - Improve grid rendering performance
   - Smooth grid transitions during zoom

### Medium Priority
3. **AI-Controlled Enemy Blobs using TensorFlow**
   - Implement neural network-based enemy AI
   - Dynamic enemy behavior patterns
   - Learning-based difficulty progression

4. **Difficulty Settings (Easy/Medium/Hard)**
   - Enemy count and aggression scaling
   - Food spawn rate adjustments
   - AI behavior complexity levels

5. **Eject Mass Functionality**
   - Allow players to eject mass for strategic purposes
   - Implement proper mass conservation
   - Add visual feedback for ejected mass

6. **Enemy Blob Consumption Mechanics**
   - Player can eat smaller enemies
   - Enemies can eat player
   - Proper collision and consumption logic

### Lower Priority
7. **Game Over Conditions**
   - Player size below minimum threshold
   - Time-based challenges
   - Achievement system

8. **Sound Effects and Music**
   - Background music
   - Sound effects for actions
   - Audio volume controls

9. **Particle Effects for Food Consumption**
   - Visual feedback when eating food
   - Growth animation effects
   - Performance-optimized particle system

## 🐛 Known Issues

- Split functionality partially working - subsequent splits after initial split need refinement
- Minor grid rendering artifacts during complex splits
- Camera zoom may occasionally be too aggressive
- Split blob positioning could be more strategic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original Agar.io game
- Built with Pygame for Python game development
- Future AI implementation planned with TensorFlow

## 📊 Development Status

- **Core Gameplay**: ✅ Complete
- **Split System**: 🔄 Partially Complete (80%)
- **AI Enemies**: ⏳ Planned
- **Sound System**: ⏳ Planned
- **Particle Effects**: ⏳ Planned

---

**Current Version**: 1.0.0  
**Last Updated**: December 2024  
**Python Version**: 3.8+  
**Pygame Version**: 2.6.0+
