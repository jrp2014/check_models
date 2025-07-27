#!/usr/bin/env python3
# Quick test of the memory formatting fix
import sys
sys.path.append(".")

from check_models import format_field_value

# Test cases for different memory value scenarios
test_cases = [
    (5.546, "5.546 GB from MLX VLM output"),  # Expected from MLX VLM
    (0.5, "0.5 MB (small value)"),           # Small MB value
    (5954972155.904, "5.95B bytes (large)"), # Large byte value
]

print("Testing format_field_value for peak_memory:")
for value, description in test_cases:
    result = format_field_value("peak_memory", value)
    print(f"  {description}")
    print(f"    Input: {value}")
    print(f"    Output: {result}")
    print(f"    Type: {type(result)}")
    print()
