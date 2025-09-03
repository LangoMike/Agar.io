# Agar.io - Single Player Edition

A single-player rendition of the popular web game Agar.io, built with Python and Pygame.

## Current Features ✅

- **Basic Game Mechanics**: Player blob movement, food consumption, size growth
- **Camera System**: Smooth camera following with world boundaries
- **Dynamic Food System**: Multiple food sizes (0.5 to 50) with realistic rarity distribution
- **Proper Scoring**: Player size IS the score (no separate score metric)
- **Collision Detection**: Proper collision between player and food
- **Petri Dish Background**: Grid pattern background similar to the original game
- **Improved UI**: Clean interface showing time and score (size)
- **Performance Optimized**: Efficient food rendering and camera calculations
- **Balanced Gameplay**: Food density allows players to gain 40-70 size in ~10 seconds

## Controls

- **Mouse Movement**: Move your blob by pointing the mouse in the desired direction
- **F11**: Toggle fullscreen mode
- **Close Window**: Click the X button to exit

## Installation & Setup

1. **Install Python 3.8+** if you haven't already
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the game**:
   ```bash
   python app.py
   ```

## Game Mechanics

- **Objective**: Grow as large as possible by consuming food pellets
- **Movement**: Your blob automatically moves toward the mouse cursor at improved speed
- **Growth**: Each food pellet adds its size value to your score (size = score)
- **Speed**: Larger blobs move slower but with better scaling (square root instead of linear)
- **World**: Large explorable world with grid pattern background

## Food System

The game features a dynamic food system with different sizes and rarity:

| Food Size | Score Value | Rarity | Visual Size |
|-----------|-------------|---------|-------------|
| 19.0      | 19.0        | 35%     | Small       |
| 25.0      | 25.0        | 25%     | Medium      |
| 30.0      | 30.0        | 20%     | Large       |
| 35.0      | 35.0        | 12%     | Huge        |
| 40.0      | 40.0        | 6%      | Massive     |
| 50.0      | 50.0        | 1.5%    | Legendary   |
| 55.0      | 55.0        | 0.5%    | Ultra-Legendary |

- **Balanced Distribution**: Smaller food is more common, larger food is rarer
- **Visual Scaling**: Food size matches its score value
- **Continuous Spawning**: New food spawns when old food is consumed

## Technical Details

- **Engine**: Pygame 2.6.1+
- **World Size**: 23040x12960 pixels (20% larger than before)
- **Screen Resolution**: 1920x1080 (resizable)
- **FPS**: 60 FPS target
- **Food Density**: 1 food per 400,000 pixels for balanced gameplay
- **Movement**: Square root scaling for better feel at all sizes
- **Architecture**: Modular design with separate Food class

## Food Mechanics

The game features intelligent food consumption rules:

- **Size Requirement**: You can only eat food that is smaller than your current size
- **Center Touch**: Food is only consumed when your blob touches the center of the food
- **Pass Through**: If you're too small, you can pass through food without consuming it
- **Strategic Growth**: You must grow strategically to access larger food pieces
- **Balanced Density**: Food is distributed evenly across the larger world

## Next Steps 🚀

- [ ] AI-controlled enemy blobs using TensorFlow
- [ ] Difficulty settings (Easy/Medium/Hard)
- [ ] Split mechanics for faster movement
- [ ] Eject mass functionality
- [ ] Enemy blob consumption mechanics
- [ ] Game over conditions
- [ ] Sound effects and music
- [ ] Particle effects for food consumption

## File Structure

```
Agar.io/
├── app.py          # Main game logic and loop
├── food.py         # Food class implementation
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Contributing

Feel free to contribute improvements or report bugs! This is a learning project to understand game development and AI implementation.
