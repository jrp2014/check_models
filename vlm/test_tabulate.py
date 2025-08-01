#!/usr/bin/env python3
"""Test script to verify tabulate integration works correctly."""

from pathlib import Path
from dataclasses import dataclass
from tabulate import tabulate
import sys

# Add the vlm directory to the path so we can import check_models
sys.path.insert(0, str(Path(__file__).parent))

from check_models import _prepare_table_data, PerformanceResult, GenerationResult

# Create some test data
@dataclass
class MockGenerationResult:
    tokens: int = 100
    prompt_tokens: int = 50
    generation_tokens: int = 50
    prompt_tps: float = 25.5
    generation_tps: float = 12.3
    peak_memory: float = 512.0
    time: float = 4.2
    text: str = "This is a test response"

# Test data
test_results = [
    PerformanceResult(
        model_name="test-model-1",
        generation=MockGenerationResult(
            tokens=120,
            prompt_tokens=60,
            generation_tokens=60,
            prompt_tps=30.2,
            generation_tps=15.1,
            peak_memory=768.0,
            time=5.5,
            text="First test response with some sample text"
        ),
        success=True,
    ),
    PerformanceResult(
        model_name="test-model-2",
        generation=MockGenerationResult(
            tokens=80,
            prompt_tokens=40,
            generation_tokens=40,
            prompt_tps=20.1,
            generation_tps=10.2,
            peak_memory=384.0,
            time=3.8,
            text="Second test response"
        ),
        success=True,
    ),
    PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="Model failed to load"
    ),
]

def test_table_generation():
    """Test that the table generation works correctly."""
    print("Testing _prepare_table_data function...")
    
    headers, rows = _prepare_table_data(test_results)
    
    print(f"Headers: {headers}")
    print(f"Number of rows: {len(rows)}")
    
    # Test HTML format
    print("\n--- HTML Format ---")
    html_table = tabulate(
        rows,
        headers=headers,
        tablefmt="unsafehtml",
    )
    print(html_table)
    
    # Test Markdown format
    print("\n--- Markdown Format ---")
    md_table = tabulate(
        rows,
        headers=headers,
        tablefmt="github",
    )
    print(md_table)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_table_generation()
