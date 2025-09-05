# Agar.io AI Training System

## Overview
This system trains AI enemies using Reinforcement Learning with PyTorch. The AI learns to eat food, grow, and survive through automated AI vs AI battles.

## How It Works

### 1. Training Process
```
AI vs AI Battles → Data Collection → Neural Network Learning → Model Updates
```

- **20 AI enemies** compete in each training episode
- **Neural network** makes decisions (movement + AI state)
- **Rewards** given for eating food, growing, and surviving
- **Models saved** every 5 episodes and when performance improves


### 2. Training Goals

## Phase 1: Basic Food Eating 
- **Target**: Consistently reach 100 size in under 20 seconds
- **Current Status**: AI can eat food and grow, but needs optimization
- **Next Steps**: Improve movement efficiency and food-seeking behavior

## Phase 2: Efficient Movement 
- Learn optimal paths to food
- Reduce time to reach food
- Improve survival rate

## Phase 3: Strategic Behavior 
- Avoid dangerous situations
- Develop food-seeking strategies
- Better resource management

## Phase 4: Advanced Tactics 
- Enemy avoidance
- Territory control
- Split mechanics

## Model Management

### What Each Model File Means
- **`best_beginner_model.pth`**: Highest performing model so far
- **`latest_beginner_model.pth`**: Most recently trained model
- **`beginner_model_episode_X.pth`**: Model at specific training episode


## Training Parameters

### Current Settings (Beginner)
- **Episodes**: 25 (reduced for faster iteration)
- **Enemies per episode**: 20
- **Max episode time**: 30 seconds
- **Target**: 100 size increase in under 20 seconds
- **Early termination**: When AI reaches target size

### Performance Metrics
- **Size growth**: How much the AI grew
- **Food eaten**: Number of food pellets consumed
- **Survival time**: How long the AI survived
- **Overall performance**: Weighted combination of all metrics

## Troubleshooting

### Common Issues
1. **AI not growing**: Check if food collision detection is working
2. **Training too slow**: Reduce episodes, increase enemies, use GPU
3. **Poor performance**: Adjust reward system, increase training episodes

## Next Steps

1. **Complete Phase 1**: Get AI to consistently reach 100 size in under 20s
2. **Increase complexity**: Add more enemies, longer episodes
3. **Intermediate training**: Move to Phase 2 once basic behavior is solid
4. **Human testing**: Play against your AI to see real progress!
