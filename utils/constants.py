"""
Game constants and configuration values
"""

# Screen and Display
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60

# World Settings
WORLD_WIDTH = int(19200 * 1.2)  # 20% larger than before
WORLD_HEIGHT = int(10800 * 1.2)

# Colors
BG_COLOR = (30, 30, 50)  # Dark blue-gray background
BG_GRADIENT_COLOR = (80, 80, 120)  # Lighter blue-gray for gradient
PLAYER_COLOR = (128, 0, 128)  # Purple
ENEMY_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
]

# UI Colors
UI_BG_COLOR = (255, 255, 255, 180)  # Semi-transparent white
UI_TEXT_COLOR = (50, 50, 50)  # Dark text
UI_BORDER_COLOR = (200, 200, 200)  # Light gray border

# Player Settings
PLAYER_START_SIZE = 20
PLAYER_MIN_SIZE = 25  # Minimum size after split
PLAYER_MAX_SPLITS = 4  # Maximum number of splits (16 total blobs)
PLAYER_SPLIT_REJOIN_TIME = 30  # Seconds before blobs rejoin
PLAYER_BASE_SPEED = 400
PLAYER_SPEED_POWER = 0.1

# Camera Settings
CAMERA_MIN_ZOOM = 0.10

# Food Settings
FOOD_SIZES = {
    10.0: 0.40,  # 40% chance - tiny food (NEW - easiest to eat)
    19.0: 0.30,  # 30% chance - small food (reduced from 35%)
    25.0: 0.20,  # 20% chance - medium food (reduced from 25%)
    30.0: 0.15,  # 15% chance - large food (reduced from 20%)
    35.0: 0.10,  # 10% chance - huge food (reduced from 12%)
    40.0: 0.04,  # 4% chance - massive food (reduced from 6%)
    50.0: 0.01,  # 1% chance - legendary food (reduced from 1.5%)
    55.0: 0.005,  # 0.5% chance - ultra-legendary food (unchanged)
}

FOOD_DENSITY = 1 / 160000  # 1 food per 160,000 pixels (increased from 175,000)
FOOD_MULTIPLIER = 0.5  # Reduced food multiplier to prevent overpopulation

# Growth System Settings
GROWTH_FACTOR = 0.3  # How quickly growth diminishes (0.1 = slow, 0.5 = fast)
MIN_GROWTH_PERCENT = 0.1  # Minimum growth as percentage of base value (10%)

# Enemy Settings
ENEMY_START_COUNT = 20  # Increased for testing - easier to find enemies
ENEMY_MIN_SIZE = 20  # Same as player start size
ENEMY_MAX_SIZE = 1400
ENEMY_SPAWN_RATE = 0.1  # New enemy per 10 seconds
ENEMY_BASE_SPEED = 300

# AI Settings
AI_UPDATE_RATE = 0.1  # Update AI every 100ms
AI_VISION_RANGE = 500  # How far enemies can see
AI_DECISION_RATE = 0.5  # How often AI makes decisions

# Difficulty Settings
DIFFICULTY_LEVELS = {
    "easy": {
        "enemy_count": 5,
        "enemy_speed_multiplier": 0.7,
        "enemy_aggression": 0.3,
        "food_spawn_rate": 1.2,
    },
    "medium": {
        "enemy_count": 10,
        "enemy_speed_multiplier": 1.0,
        "enemy_aggression": 0.6,
        "food_spawn_rate": 1.0,
    },
    "hard": {
        "enemy_count": 15,
        "enemy_speed_multiplier": 1.3,
        "enemy_aggression": 0.9,
        "food_spawn_rate": 0.8,
    },
}

# Collision Settings
COLLISION_PRECISION = 0.8  # How precise collision detection is
FOOD_CENTER_REQUIREMENT = True  # Must touch center to eat food

# Minimap Settings
MINIMAP_SIZE = 200
MINIMAP_OPACITY = 150

# Sound Settings
SOUND_ENABLED = True
MUSIC_ENABLED = True
SOUND_VOLUME = 0.7
MUSIC_VOLUME = 0.5

# Performance Settings
MAX_PARTICLES = 1000
PARTICLE_LIFETIME = 2.0  # seconds
ENABLE_PARTICLES = True
ENABLE_SHADOWS = False  # Future feature

# Game States
GAME_STATE_MENU = "menu"
GAME_STATE_PLAYING = "playing"
GAME_STATE_PAUSED = "paused"
GAME_STATE_GAME_OVER = "game_over"
GAME_STATE_VICTORY = "victory"
