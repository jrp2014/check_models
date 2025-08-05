#!/usr/bin/env python3
"""Quick test to verify timing attribute handling in check_models.py."""

import sys
from pathlib import Path

# Add the vlm directory to path
sys.path.insert(0, str(Path(__file__).parent))

from check_models import PerformanceResult, _sort_results_by_time
from mlx_vlm.generate import GenerationResult

def test_timing_attributes():
    """Test that we can add timing attributes and sort by them."""
    
    # Create mock GenerationResult objects
    gen1 = GenerationResult(
        text="Test response 1",
        token=None,
        logprobs=None,
        prompt_tokens=10,
        generation_tokens=20,
        total_tokens=30,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=1000,
    )
    
    gen2 = GenerationResult(
        text="Test response 2", 
        token=None,
        logprobs=None,
        prompt_tokens=15,
        generation_tokens=25,
        total_tokens=40,
        prompt_tps=110.0,
        generation_tps=60.0,
        peak_memory=1200,
    )
    
    # Create PerformanceResult objects with timing data in the dataclass
    result1 = PerformanceResult(
        model_name="test-model-1",
        success=True,
        generation=gen1,
        error_stage=None,
        error_message=None,
        captured_output_on_fail=None,
        elapsed_time=1.5,  # Slower
        model_load_time=0.8,
        total_time=2.3,
    )
    
    result2 = PerformanceResult(
        model_name="test-model-2", 
        success=True,
        generation=gen2,
        error_stage=None,
        error_message=None,
        captured_output_on_fail=None,
        elapsed_time=0.8,  # Faster
        model_load_time=0.5,
        total_time=1.3,
    )
    
    results = [result1, result2]
    print(f"Before sorting:")
    for i, r in enumerate(results):
        print(f"  result{i+1}: {r.model_name}, elapsed_time={r.elapsed_time}")
    
    # Test sorting
    sorted_results = _sort_results_by_time(results)
    
    print(f"After sorting:")
    for i, r in enumerate(sorted_results):
        print(f"  result{i+1}: {r.model_name}, elapsed_time={r.elapsed_time}")
    
    # Check if sorting worked correctly (gen2 should be first since it's faster)
    if sorted_results[0].model_name != "test-model-2":
        raise ValueError(f"Expected test-model-2 first, got {sorted_results[0].model_name}")
    if sorted_results[1].model_name != "test-model-1":
        raise ValueError(f"Expected test-model-1 second, got {sorted_results[1].model_name}")
    
    print("✅ Test passed! Timing attributes and sorting work correctly.")

if __name__ == "__main__":
    test_timing_attributes()
