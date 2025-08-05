#!/usr/bin/env python3

"""Test the new multi-line console table formatting."""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Test the formatting functions
from check_models import _format_model_name_multiline, _format_output_multiline

def test_formatting():
    """Test the formatting functions."""
    print("Testing model name formatting:")
    
    # Test cases for model names
    test_names = [
        "short",
        "microsoft/Phi-3.5-vision-instruct", 
        "Qwen/Qwen2-VL-7B-Instruct",
        "very-long-model-name-that-should-be-split",
        "microsoft/very-long-and-complex-model-name-for-testing"
    ]
    
    for name in test_names:
        formatted = _format_model_name_multiline(name)
        print(f"'{name}' -> '{formatted}'")
        print("---")
    
    print("\nTesting output formatting:")
    
    # Test cases for output text
    test_outputs = [
        "Short output",
        "This is a medium length output that should fit reasonably well",
        "This is a very long output text that should definitely be wrapped to multiple lines because it exceeds the maximum length we want to display in a single line of the console table format",
        "Error: AttributeError occurred during model loading - this is a typical error message that might appear"
    ]
    
    for output in test_outputs:
        formatted = _format_output_multiline(output)
        print(f"'{output}' -> \n'{formatted}'")
        print("---")

if __name__ == "__main__":
    test_formatting()
