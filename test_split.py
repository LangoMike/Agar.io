




"""
Test script to verify split functionality
"""

import pygame
from pygame.math import Vector2
from entities.player import Player
from entities.split_blob import SplitBlob
from utils.constants import PLAYER_START_SIZE

def test_split_functionality():
    """Test the split functionality"""
    print("🧪 Testing split functionality...")
    
    # Initialize pygame
    pygame.init()
    
    # Create a player
    player = Player(Vector2(100, 100), 60)  # Start with size 60 to allow splitting
    print(f"✅ Player created with size: {player.size}")
    
    # Test initial state
    print(f"📊 Initial split count: {player.split_count}")
    print(f"🔀 Can split: {player.can_split()}")
    print(f"📦 Is split: {player.is_split}")
    print(f"📦 Split blobs: {len(player.split_blobs)}")
    
    # Test first split
    print("\n🔄 Testing first split...")
    if player.split():
        print(f"✅ Split successful! Split count: {player.split_count}")
        print(f"📦 Split blobs: {len(player.split_blobs)}")
        print(f"📊 Player size after split: {player.size}")
        
        for i, blob in enumerate(player.split_blobs):
            print(f"  📦 Blob {i+1}: size={blob.size}, active={blob.is_active}")
    else:
        print("❌ Split failed!")
    
    # Test second split
    print("\n🔄 Testing second split...")
    if player.split():
        print(f"✅ Second split successful! Split count: {player.split_count}")
        print(f"📦 Split blobs: {len(player.split_blobs)}")
        print(f"📊 Player size after split: {player.size}")
        
        for i, blob in enumerate(player.split_blobs):
            print(f"  📦 Blob {i+1}: size={blob.size}, active={blob.is_active}")
    else:
        print("❌ Second split failed!")
    
    # Test split blob positioning
    print("\n📍 Testing split blob positioning...")
    for i, blob in enumerate(player.split_blobs):
        print(f"  📦 Blob {i+1}: pos=({blob.position.x:.1f}, {blob.position.y:.1f})")
        print(f"     offset=({blob.split_offset.x:.1f}, {blob.split_offset.y:.2f})")
        print(f"     angle={blob.split_angle:.2f}")
    
    # Test collision avoidance
    print("\n🚫 Testing collision avoidance...")
    for i, blob in enumerate(player.split_blobs):
        other_blobs = [b for b in player.split_blobs if b != blob]
        avoidance = blob._calculate_avoidance_movement(other_blobs)
        print(f"  📦 Blob {i+1}: avoidance=({avoidance.x:.2f}, {avoidance.y:.2f})")
    
    pygame.quit()
    print("\n✅ Split functionality test completed!")

if __name__ == "__main__":
    test_split_functionality()
