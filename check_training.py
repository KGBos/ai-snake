#!/usr/bin/env python3
"""Check if DQN training is working by analyzing saved models."""

import torch
import numpy as np
import os

def check_model(model_path):
    """Check if a trained model shows signs of learning."""
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        return False
    
    try:
        model = torch.load(model_path, map_location='cpu')
        print(f"✅ Model '{model_path}' loaded successfully!")
        
        # Check key metrics
        epsilon = model.get('epsilon', None)
        training_step = model.get('training_step', None)
        episode_rewards = model.get('episode_rewards', [])
        episode_lengths = model.get('episode_lengths', [])
        
        print(f"\n📊 Training Statistics:")
        print(f"  Epsilon: {epsilon:.3f}" if epsilon else "  Epsilon: N/A")
        print(f"  Training steps: {training_step}" if training_step else "  Training steps: N/A")
        print(f"  Episodes recorded: {len(episode_rewards)}")
        
        if episode_rewards:
            recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            early_rewards = episode_rewards[:50] if len(episode_rewards) >= 50 else episode_rewards
            
            print(f"\n🎯 Learning Progress:")
            print(f"  Early episodes (avg reward): {np.mean(early_rewards):.2f}")
            print(f"  Recent episodes (avg reward): {np.mean(recent_rewards):.2f}")
            print(f"  Best reward: {max(episode_rewards):.2f}")
            print(f"  Worst reward: {min(episode_rewards):.2f}")
            
            # Check if learning is happening
            if len(episode_rewards) >= 20:
                improvement = np.mean(recent_rewards) - np.mean(early_rewards)
                print(f"  Improvement: {improvement:+.2f}")
                
                if improvement > 0:
                    print("✅ Training is working! Recent performance is better than early performance.")
                else:
                    print("⚠️  Training may not be working effectively yet.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def compare_models():
    """Compare different models to see learning progress."""
    models = [
        "demo_model.pth",
        "watchtrain_model_final.pth",
        "snake_dqn_model.pth"
    ]
    
    print("🔍 Comparing available models:")
    for model in models:
        if os.path.exists(model):
            print(f"\n--- {model} ---")
            check_model(model)
        else:
            print(f"\n❌ {model} not found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        check_model(sys.argv[1])
    else:
        compare_models() 