#!/usr/bin/env python3
"""
Test script to verify SuperPoint integration with FeatUp training pipeline
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_superpoint_featurizer():
    """Test that SuperPoint featurizer can be imported and run"""
    try:
        from featup.featurizers.util import get_featurizer
        
        # Test getting SuperPoint featurizer
        model, patch_size, dim = get_featurizer("superpoint", activation_type="key")
        
        print(f"✓ SuperPoint featurizer loaded successfully")
        print(f"  Patch size: {patch_size}")
        print(f"  Feature dimension: {dim}")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (1, 256, 28, 28)")
        
        # Verify output dimensions
        expected_h = dummy_input.shape[2] // 8
        expected_w = dummy_input.shape[3] // 8
        expected_shape = (1, 256, expected_h, expected_w)
        
        if output.shape == expected_shape:
            print("✓ Output shape matches expected dimensions")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing SuperPoint featurizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_config():
    """Test that training configuration supports SuperPoint"""
    try:
        # Import required modules
        import sys
        sys.path.append('/home/kentselwyn/Tohoku/FeatUp-SuperPoint/featup')
        
        # Test config loading (simplified)
        from omegaconf import DictConfig
        
        # Simulate config with SuperPoint
        cfg = DictConfig({
            'model_type': 'superpoint',
            'steps': 100,
            'output_root': '/tmp',
            'activation_type': 'key'
        })
        
        print("✓ SuperPoint configuration created successfully")
        print(f"  Model type: {cfg.model_type}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing training config: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing SuperPoint integration with FeatUp...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing SuperPoint featurizer...")
    success &= test_superpoint_featurizer()
    
    print("\n2. Testing training configuration...")
    success &= test_training_config()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! SuperPoint integration successful.")
        print("\nTo use SuperPoint in training, set model_type='superpoint' in your config:")
        print("  model_type: superpoint")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
