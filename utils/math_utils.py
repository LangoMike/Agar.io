"""
Mathematical utilities for the game
"""

import math
from pygame.math import Vector2
from typing import Tuple, List, Optional


def distance_between_points(point1: Vector2, point2: Vector2) -> float:
    """Calculate distance between two points"""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def angle_between_points(point1: Vector2, point2: Vector2) -> float:
    """Calculate angle between two points in radians"""
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return math.atan2(dy, dx)


def normalize_vector(vector: Vector2) -> Vector2:
    """Normalize a vector to unit length"""
    length = vector.length()
    if length == 0:
        return Vector2(0, 0)
    return vector / length


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(value, max_val))


def clamp_vector(vector: Vector2, max_length: float) -> Vector2:
    """Clamp vector length to maximum"""
    if vector.length() <= max_length:
        return vector
    return normalize_vector(vector) * max_length


def point_in_circle(
    point: Vector2, circle_center: Vector2, circle_radius: float
) -> bool:
    """Check if a point is inside a circle"""
    return distance_between_points(point, circle_center) <= circle_radius


def circles_intersect(
    center1: Vector2, radius1: float, center2: Vector2, radius2: float
) -> bool:
    """Check if two circles intersect"""
    distance = distance_between_points(center1, center2)
    return distance <= (radius1 + radius2)


def circle_rect_intersect(circle_center: Vector2, circle_radius: float, rect) -> bool:
    """Check if a circle intersects with a rectangle"""
    # Find the closest point to the circle within the rectangle
    closest_x = clamp_value(circle_center.x, rect.left, rect.right)
    closest_y = clamp_value(circle_center.y, rect.top, rect.bottom)

    # Calculate the distance between the circle's center and this closest point
    distance = distance_between_points(circle_center, Vector2(closest_x, closest_y))

    # If the distance is less than the circle's radius, an intersection occurs
    return distance <= circle_radius


def get_random_point_in_circle(center: Vector2, radius: float) -> Vector2:
    """Get a random point inside a circle"""
    import random

    angle = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.uniform(0, 1))
    return Vector2(center.x + r * math.cos(angle), center.y + r * math.sin(angle))


def get_random_point_in_rect(rect) -> Vector2:
    """Get a random point inside a rectangle"""
    import random

    return Vector2(
        random.uniform(rect.left, rect.right), random.uniform(rect.top, rect.bottom)
    )


def lerp(start: float, end: float, factor: float) -> float:
    """Linear interpolation between two values"""
    return start + (end - start) * factor


def lerp_vector(start: Vector2, end: Vector2, factor: float) -> Vector2:
    """Linear interpolation between two vectors"""
    return Vector2(lerp(start.x, end.x, factor), lerp(start.y, end.y, factor))


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Smooth step function for smooth transitions"""
    t = clamp_value((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def calculate_zoom_factor(
    player_size: float,
    base_size: float = 20,
    min_zoom: float = 0.15,
    power: float = 0.2,
) -> float:
    """Calculate camera zoom factor based on player size"""
    return max(min_zoom, (base_size / player_size) ** power)


def world_to_screen(
    world_pos: Vector2, camera_x: float, camera_y: float, zoom_factor: float
) -> Vector2:
    """Convert world coordinates to screen coordinates"""
    return Vector2(
        (world_pos.x - camera_x) * zoom_factor, (world_pos.y - camera_y) * zoom_factor
    )


def screen_to_world(
    screen_pos: Vector2, camera_x: float, camera_y: float, zoom_factor: float
) -> Vector2:
    """Convert screen coordinates to world coordinates"""
    return Vector2(
        (screen_pos.x / zoom_factor) + camera_x, (screen_pos.y / zoom_factor) + camera_y
    )


def calculate_effective_screen_dimensions(
    screen_width: int, screen_height: int, zoom_factor: float
) -> Tuple[int, int]:
    """Calculate effective screen dimensions with zoom"""
    return (int(screen_width / zoom_factor), int(screen_height / zoom_factor))


def is_point_in_screen(point: Vector2, screen_width: int, screen_height: int) -> bool:
    """Check if a point is within screen bounds"""
    return 0 <= point.x <= screen_width and 0 <= point.y <= screen_height


def get_direction_vector(start: Vector2, end: Vector2) -> Vector2:
    """Get normalized direction vector from start to end"""
    direction = end - start
    return normalize_vector(direction)


def rotate_vector(vector: Vector2, angle: float) -> Vector2:
    """Rotate a vector by an angle in radians"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Vector2(
        vector.x * cos_a - vector.y * sin_a, vector.x * sin_a + vector.y * cos_a
    )


def calculate_growth_value(
    base_value: float, current_size: float, growth_factor: float = None
) -> float:
    """
    Calculate growth value with diminishing returns based on current size.

    Args:
        base_value: The base growth value (e.g., food score or enemy mass)
        current_size: Current size of the blob
        growth_factor: How quickly growth diminishes (0.1 = slow, 0.5 = fast)

    Returns:
        The actual growth value to apply
    """
    from utils.constants import GROWTH_FACTOR, MIN_GROWTH_PERCENT

    # Use default growth factor if none provided
    if growth_factor is None:
        growth_factor = GROWTH_FACTOR

    # Use a logarithmic decay function: growth = base_value / (1 + growth_factor * log(current_size/20))
    # This means:
    # - At size 20: growth = base_value (100% growth)
    # - At size 200: growth ≈ base_value / 1.3 (77% growth)
    # - At size 2000: growth ≈ base_value / 2.0 (50% growth)
    # - At size 20000: growth ≈ base_value / 2.7 (37% growth)

    if current_size <= 20:
        return base_value

    # Calculate diminishing factor
    size_ratio = current_size / 20.0
    diminishing_factor = 1 + growth_factor * math.log(size_ratio)

    # Apply diminishing returns
    actual_growth = base_value / diminishing_factor

    # Ensure minimum growth (at least MIN_GROWTH_PERCENT of base value)
    min_growth = base_value * MIN_GROWTH_PERCENT
    return max(actual_growth, min_growth)
