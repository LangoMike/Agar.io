import pygame
import random
from sys import exit
from pygame.math import Vector2
from food import Food

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # Window size
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Solo-player Agar.io")

# Colors
BG_COLOR = (240, 248, 255)  # Light blue background like petri dish
GRID_COLOR = (200, 220, 240)  # Subtle grid lines
PLAYER_COLOR = (128, 0, 128)  # Purple
UI_BG_COLOR = (255, 255, 255, 180)  # Semi-transparent white
UI_TEXT_COLOR = (50, 50, 50)  # Dark text

# World settings
WORLD_WIDTH, WORLD_HEIGHT = int(19200 * 1.2), int(10800 * 1.2)  # 20% larger world
world = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
world.fill(BG_COLOR)

# Draw petri dish grid pattern
GRID_SIZE = 100
for x in range(0, WORLD_WIDTH, GRID_SIZE):
    pygame.draw.line(world, GRID_COLOR, (x, 0), (x, WORLD_HEIGHT), 1)
for y in range(0, WORLD_HEIGHT, GRID_SIZE):
    pygame.draw.line(world, GRID_COLOR, (0, y), (WORLD_WIDTH, y), 1)

# Camera setup
clock = pygame.time.Clock()

# Player settings
player_head_pos = Vector2(WORLD_WIDTH / 2, WORLD_HEIGHT / 2)
player_radi = 20  # Starting size/score

# Time variable
time = 0

# Food generation settings
# Food sizes and their probabilities (size: probability)
FOOD_SIZES = {
    19.0: 0.35,   # 35% chance - small food
    25.0: 0.25,   # 25% chance - medium food
    30.0: 0.20,   # 20% chance - large food
    35.0: 0.12,   # 12% chance - huge food
    40.0: 0.06,   # 6% chance - massive food
    50.0: 0.015,  # 1.5% chance - legendary food
    55.0: 0.005   # 0.5% chance - ultra-legendary food
}

# Calculate total food needed for good gameplay
# Target: player should gain 40-70 size in ~10 seconds
# With average food value of ~30, we need balanced food density
WORLD_AREA = WORLD_WIDTH * WORLD_HEIGHT
FOOD_DENSITY = 1 / 50000  # 1 food per 50,000 pixels
maxFood = int(WORLD_AREA * FOOD_DENSITY * 1.1 / 8)  # 1/8th of the previous amount
food_group = pygame.sprite.Group()

# Methods
def random_color(background_color):
    while True:
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        color = (r, g, b)

        if color != background_color:
            ret_color = pygame.Color(r, g, b)
            return ret_color

def random_center():
    while True:
        x, y = random.randint(0, WORLD_WIDTH), random.randint(0, WORLD_HEIGHT)
        center = (x, y)
        return center

def generate_food_size():
    """Generate food size based on probability distribution"""
    rand = random.random()
    cumulative_prob = 0
    
    for size, prob in FOOD_SIZES.items():
        cumulative_prob += prob
        if rand <= cumulative_prob:
            return size
    
    return 0.5  # Fallback to smallest size

def draw_petri_dish_background(surface, camera_x, camera_y, zoom_factor):
    """Draw the petri dish background with grid pattern and proper zoom scaling"""
    # Fill background
    surface.fill(BG_COLOR)
    
    # Calculate grid spacing that scales with zoom
    # At zoom 1.0, grid lines are 100 pixels apart
    # At zoom 2.0, grid lines are 200 pixels apart (zoomed out)
    scaled_grid_size = int(GRID_SIZE * zoom_factor)
    
    # Calculate visible grid area
    start_x = int((camera_x // scaled_grid_size) * scaled_grid_size)
    start_y = int((camera_y // scaled_grid_size) * scaled_grid_size)
    
    # Draw vertical grid lines
    for x in range(start_x, start_x + int(SCREEN_WIDTH / zoom_factor) + scaled_grid_size, scaled_grid_size):
        if 0 <= x <= WORLD_WIDTH:
            screen_x = int((x - camera_x) * zoom_factor)
            pygame.draw.line(surface, GRID_COLOR, 
                           (screen_x, 0), (screen_x, SCREEN_HEIGHT), 1)
    
    # Draw horizontal grid lines
    for y in range(start_y, start_y + int(SCREEN_HEIGHT / zoom_factor) + scaled_grid_size, scaled_grid_size):
        if 0 <= y <= WORLD_HEIGHT:
            screen_y = int((y - camera_y) * zoom_factor)
            pygame.draw.line(surface, GRID_COLOR, 
                           (0, screen_y), (SCREEN_WIDTH, screen_y), 1)

def draw_ui(surface, time, player_radi, zoom_factor=1.0):
    """Draw improved UI with better styling - size IS the score"""
    # Create UI background
    ui_surface = pygame.Surface((300, 80), pygame.SRCALPHA)
    ui_surface.fill(UI_BG_COLOR)
    
    # Draw rounded rectangle effect
    pygame.draw.rect(ui_surface, (255, 255, 255), (0, 0, 300, 80), border_radius=10)
    pygame.draw.rect(ui_surface, (200, 200, 200), (0, 0, 300, 80), 2, border_radius=10)
    
    # Font setup
    font_large = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    # Time display
    time_hours = int(time / 3600000)
    time_minutes = int(time / 60000)
    time_seconds = int(time / 1000)
    time_notation = (
        f"{time_hours:02d}:"
        f"{int(time_minutes - (time_hours * 60)):02d}:"
        f"{int(time_seconds - (time_minutes * 60)):02d}"
    )
    
    time_text = font_large.render(f"Time: {time_notation}", True, UI_TEXT_COLOR)
    score_text = font_large.render(f"Score: {player_radi:.1f}", True, UI_TEXT_COLOR)
    zoom_text = font_small.render(f"Zoom: {zoom_factor:.2f}x", True, UI_TEXT_COLOR)
    
    # Position text
    ui_surface.blit(time_text, (10, 10))
    ui_surface.blit(score_text, (10, 35))
    ui_surface.blit(zoom_text, (10, 55))
    
    # Draw UI to screen
    surface.blit(ui_surface, (10, 10))

def draw_minimap(surface, player_pos, camera_x, camera_y, zoom_factor):
    """Draw a minimap in the bottom-left corner showing player position"""
    minimap_size = 200
    minimap_surface = pygame.Surface((minimap_size, minimap_size), pygame.SRCALPHA)
    minimap_surface.fill((255, 255, 255, 150))  # Semi-transparent white
    
    # Draw world boundaries on minimap
    pygame.draw.rect(minimap_surface, (100, 100, 100), (0, 0, minimap_size, minimap_size), 2)
    
    # Calculate player position on minimap (world coordinates to minimap coordinates)
    # Player should be at center of minimap when at center of world
    player_minimap_x = int((player_pos.x / WORLD_WIDTH) * minimap_size)
    player_minimap_y = int((player_pos.y / WORLD_HEIGHT) * minimap_size)
    
    # Draw player as red square on minimap
    pygame.draw.rect(minimap_surface, (255, 0, 0), 
                    (player_minimap_x - 3, player_minimap_y - 3, 6, 6))
    
    # Draw camera viewport on minimap (showing what area is currently visible)
    viewport_left = int((camera_x / WORLD_WIDTH) * minimap_size)
    viewport_top = int((camera_y / WORLD_HEIGHT) * minimap_size)
    viewport_width = int((SCREEN_WIDTH / zoom_factor / WORLD_WIDTH) * minimap_size)
    viewport_height = int((SCREEN_HEIGHT / zoom_factor / WORLD_HEIGHT) * minimap_size)
    
    # Draw viewport rectangle
    pygame.draw.rect(minimap_surface, (0, 0, 255, 100), 
                    (viewport_left, viewport_top, viewport_width, viewport_height), 1)
    
    # Blit minimap to bottom-left corner
    surface.blit(minimap_surface, (10, SCREEN_HEIGHT - minimap_size - 10))

# Add food to food group with different sizes
for _ in range(maxFood):
    food_size = generate_food_size()
    food_group.add(
        Food(
            world,
            random_color(BG_COLOR),
            random_center(),
            food_size,
        )
    )

########## GAME LOOP ##########
running = True
while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()

    # Calculate player speed based on size - much faster now
    player_speed = 400 / max(1, player_radi ** 0.2)  # root scaling for smoothr feel

    # Smooth Camera Zoom System
    # Calculate zoom factor based on player size
    # Base zoom: at size 20, zoom = 1.0 (normal view)
    # As player grows, zoom DECREASES subtly to see slightly more of the world
    BASE_SIZE = 20  # Reference size for zoom calculation
    # Use a small power (0.2) to make zoom changes subtle but noticeable
    zoom_factor = max(0.15, (BASE_SIZE / player_radi) ** 0.2)  # Minimum zoom of 0.15x
    
    # This means:
    # Size 20 = 1.0x zoom (normal view)
    # Size 30 = 0.92x zoom (see 1.09x more world - subtle but noticeable)
    # Size 50 = 0.86x zoom (see 1.16x more world - gentle zoom out)
    # Size 100 = 0.76x zoom (see 1.32x more world - moderate zoom out)
    # Size 200 = 0.68x zoom (see 1.47x more world - noticeable zoom out)
    
    # Calculate effective screen dimensions with zoom
    effective_screen_width = int(SCREEN_WIDTH / zoom_factor)
    effective_screen_height = int(SCREEN_HEIGHT / zoom_factor)

    # Camera Movement with zoom
    camera_x = player_head_pos.x - effective_screen_width // 2
    camera_y = player_head_pos.y - effective_screen_height // 2 
    
    # Clamp camera to prevent showing areas outside world
    camera_x = max(0, min(camera_x, WORLD_WIDTH - effective_screen_width))
    camera_y = max(0, min(camera_y, WORLD_HEIGHT - effective_screen_height))

    # Draw background with proper zoom scaling
    draw_petri_dish_background(screen, camera_x, camera_y, zoom_factor)

    # Move Player towards mouse
    mouse_x, mouse_y = pygame.mouse.get_pos()
    cursor_pos = Vector2(mouse_x, mouse_y)
    
    # Convert mouse position to world coordinates (accounting for zoom)
    world_mouse_x = (cursor_pos.x / zoom_factor) + camera_x
    world_mouse_y = (cursor_pos.y / zoom_factor) + camera_y
    world_mouse_pos = Vector2(world_mouse_x, world_mouse_y)
    
    # Move player towards mouse in world coordinates
    direction = world_mouse_pos - player_head_pos
    if direction.length() > 0:
        direction = direction.normalize()
        player_head_pos += direction * player_speed * clock.get_time() / 1000

    # Draw player (relative to camera position with zoom)
    player_screen_x = (player_head_pos.x - camera_x) * zoom_factor
    player_screen_y = (player_head_pos.y - camera_y) * zoom_factor
    pygame.draw.circle(
        screen,
        PLAYER_COLOR,
        (int(player_screen_x), int(player_screen_y)),
        int(player_radi * zoom_factor),
    )

    # Food Collision Detection (using world coordinates)
    # Only eat food if player is bigger than food AND touches the center
    
    # Collect food to remove and new food to add
    food_to_remove = []
    new_food_to_add = []
    
    for food in food_group:
        # Calculate distance from player center to food center
        distance_to_food_center = ((player_head_pos.x - food.rect.centerx) ** 2 + 
                                  (player_head_pos.y - food.rect.centery) ** 2) ** 0.5
        
        # Check if player is big enough to eat this food AND touches the center
        if (player_radi > food.size and  # Player must be bigger than food
            distance_to_food_center <= player_radi):  # Player must touch food center
            
            # Add the food's score value to player size
            player_radi += food.score_value
            
            # Mark food for removal and create replacement
            food_to_remove.append(food)
            new_food_size = generate_food_size()
            new_food_to_add.append(
                Food(
                    world,
                    random_color(BG_COLOR),
                    random_center(),
                    new_food_size,
                )
            )
    
    # Process food changes (avoid modifying group while iterating)
    for food in food_to_remove:
        food_group.remove(food)
    
    for new_food in new_food_to_add:
        food_group.add(new_food)

    # Draw Food with camera offset and zoom
    for food in food_group:
        food.draw(screen, camera_x, camera_y, zoom_factor)

    # Draw UI
    draw_ui(screen, time, player_radi, zoom_factor)
    draw_minimap(screen, player_head_pos, camera_x, camera_y, zoom_factor)

    # flip() the display to put work on screen
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 60
    time += clock.get_time()

pygame.quit()
