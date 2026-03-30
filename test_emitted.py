#!/usr/bin/env python3
"""Test the actual emitted Python code from nd decompile."""
import subprocess
import json
import sys

# Generate decompiled code
result = subprocess.run(
    ['./target/release/nd', 'decompile', 'examples/parity_transformer.json'],
    capture_output=True, text=True
)

# Parse out the Python code (skip status lines at the start)
lines = result.stdout.split('\n')
py_start = 0
for i, line in enumerate(lines):
    if line.startswith('import math'):
        py_start = i
        break

code = '\n'.join(lines[py_start:])

# Execute the decompiled code
exec(code)

# Load tests
with open('examples/parity_transformer_tests.json') as f:
    tests = json.load(f)

def predict(logits):
    final_logits = logits[-1]
    return final_logits.index(max(final_logits))

print('Testing EMITTED Python code from nd decompile:')
print('=' * 60)
passed = 0
for i, test in enumerate(tests):
    tokens = test['tokens']
    expected = test['expected']
    logits = decompiled(tokens)
    pred = predict(logits)
    status = 'PASS' if pred == expected else 'FAIL'
    if pred == expected:
        passed += 1
    print(f'Test {i+1}: tokens={tokens}, expected={expected}, pred={pred} [{status}]')

print('=' * 60)
print(f'Result: {passed}/{len(tests)} passed ({100*passed/len(tests):.0f}%)')

if passed == len(tests):
    print('✓ PERFECT — emitted code matches all test cases')
    sys.exit(0)
else:
    print('✗ FAILED — emitted code does not match all test cases')
    print(f'  The quantization precision loss causes {len(tests) - passed} failures')
    sys.exit(1)
