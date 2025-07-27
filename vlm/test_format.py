#!/usr/bin/env python3
# Test format_field_value function
import sys
sys.path.append('.')

from check_models import format_field_value, MB_CONVERSION

# Test with a realistic peak memory value (5.546 GB = ~5.95 billion bytes)
test_bytes = 5.546 * 1024 * 1024 * 1024  # 5.546 GB in bytes
print(f'Input: {test_bytes} bytes')
result = format_field_value('peak_memory', test_bytes)
print(f'format_field_value result: {result}')
print(f'Type: {type(result)}')

# Test with a small value that might represent what MLX VLM actually returns
small_value = 5.546  # Maybe MLX VLM returns GB directly?
result2 = format_field_value('peak_memory', small_value)
print(f'Small value {small_value} -> {result2}')
