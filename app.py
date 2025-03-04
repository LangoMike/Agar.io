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
BG_COLOR = (50, 50, 50)  # Game world color

# World settings
WORLD_WIDTH, WORLD_HEIGHT = 19200, 10800  # Large world size
world = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
world.fill("black")
WORLD_X = (WORLD_WIDTH - SCREEN_WIDTH) // 2  # Center world horizontally
WORLD_Y = (WORLD_HEIGHT - SCREEN_HEIGHT) // 2  # Center world vertically

# Camera setup
clock = pygame.time.Clock()
clock_surface = pygame.Surface((200, 25))
clock_surface.fill("White")
score_surface = pygame.Surface((150, 25))
score_surface.fill("White")

# Player settings
PLAYER_COLOR = (128, 0, 128)  # Purple

# Time and Score variables
time = 0
score = 0
# player orb creation
player_head_pos = pygame.Vector2(WORLD_WIDTH / 2, WORLD_HEIGHT / 2)
player_radi = 20

# Generate food list + max amount of food
maxFood = WORLD_HEIGHT * WORLD_WIDTH / 10000
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


# Add food to food group
for _ in range(int(maxFood)):
    food_group.add(
        Food(
            world,
            random_color(BG_COLOR),
            random_center(),
            5,
        )
    )

########## GAME LOOP ##########
running = True
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.display.toggle_fullscreen:
            fullscreen = True
        if event.type == (pygame.mouse.get_pressed() == True):
            print("Mouse is pressed")
            speed *= 2
        else:
            speed = 200 / player_radi

    # Create text
    text_font = pygame.font.Font(None, 30)
    time_hours = int(time / 3600000)
    time_minutes = int(time / 60000)
    time_seconds = int(time / 1000)
    time_notation = (
        str(time_hours)
        + ":"
        + str(int(time_minutes - (time_hours * 60)))
        + ":"
        + str(int(time_seconds - (time_minutes * 60)))
    )
    clock_text_surface = text_font.render("Time: " + time_notation, True, "Black")
    score_text_surface = text_font.render("Score: " + str(score), True, "Black")

    # Add surfaces and text for clock/score
    screen.fill(BG_COLOR)
    screen.blit(clock_surface, (0, 0))
    screen.blit(clock_text_surface, (0, 0))
    screen.blit(score_surface, (SCREEN_WIDTH - 150, 0))
    screen.blit(score_text_surface, ((SCREEN_WIDTH - 150, 0)))

    # RENDER YOUR GAME HERE

    # Visable portion of world
    # camera_x, camera_y = player_head_pos.x - screen.get_width() // 2, player_head_pos.y - screen.get_height() // 2
    # for food in food_group:
    #     if (camera_x <= food.rect.centerx <= camera_x + screen.get_width() and
    #         camera_y <= food.rect.centery <= camera_y + screen.get_height()):
    #         screen.blit(food.image, (food.rect.centerx - camera_x, food.rect.centery - camera_y))

    # Camera Movement
    camera_x, camera_y = (
        player_head_pos.x - SCREEN_WIDTH // 2,
        player_head_pos.y - SCREEN_HEIGHT // 2,
    )
    # Clamp camera to prevent showing areas outside world
    camera_x = max(0, min(camera_x, WORLD_WIDTH - SCREEN_WIDTH))
    camera_y = max(0, min(camera_y, WORLD_HEIGHT - SCREEN_HEIGHT))

    # Draw player (relative to camera position)
    player_screen_x = player_head_pos.x - camera_x
    player_screen_y = player_head_pos.y - camera_y
    player_rect = pygame.draw.circle(
        screen,
        PLAYER_COLOR,
        (int(player_screen_x), int(player_screen_y)),
        player_radi,
    )

    # Move Player
    # Mouse Location
    mouse_x, mouse_y = pygame.mouse.get_pos()
    cursor_pos = Vector2(mouse_x, mouse_y)

    # Move Snake towards mouse location
    player_head_pos = player_head_pos.move_towards(cursor_pos, speed)
    player_rect.center = player_head_pos

    # Food Collision Detection
    for food in food_group:
        if player_rect.colliderect(food):
            food_group.remove(food)
            food_group.add(
                Food(
                    world,
                    random_color(BG_COLOR),
                    random_center(),
                    5,
                )
            )
            player_radi += 0.4
            player_rect.update(player_rect)
            score += 1
            print(player_radi)

    # Update and Draw Food
    food_group.update()
    food_group.draw(screen, BG_COLOR)

    # flip() the display to put work on screen
    pygame.display.flip()
    pygame.display.update()
    clock.tick(60)  # limits FPS to 60
    time += clock.get_time()
pygame.quit()
