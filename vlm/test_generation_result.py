#!/usr/bin/env python3
"""Quick test to see GenerationResult fields."""

try:
    from mlx_vlm.generate import GenerationResult
    from dataclasses import fields
    
    print("GenerationResult fields:")
    for field in fields(GenerationResult):
        print(f"  {field.name}: {field.type}")
        
    print("\nDone.")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")
