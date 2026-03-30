#!/usr/bin/env python3
"""Test emitted Python code from nd decompile for RNNs."""
import subprocess
import json
import sys

tests = [
    ("examples/parity3.json", "examples/parity3_tests.json", "parity3"),
    ("examples/regex/contains_11.json", "examples/regex/contains_11_tests.json", "contains_11"),
    ("examples/regex/divisible_by_3.json", "examples/regex/divisible_by_3_tests.json", "divisible_by_3"),
    ("examples/mod3_add.json", "examples/mod3_add_tests.json", "mod3_add"),
]

all_passed = True

for weights_file, tests_file, name in tests:
    # Generate decompiled code
    result = subprocess.run(
        ['./target/release/nd', 'decompile', weights_file],
        capture_output=True, text=True
    )
    
    # Parse out the Python code
    lines = result.stdout.split('\n')
    py_start = 0
    for i, line in enumerate(lines):
        if line.startswith('def '):
            py_start = i
            break
    
    code = '\n'.join(lines[py_start:])
    exec(code)
    
    # Load tests
    with open(tests_file) as f:
        test_cases = json.load(f)
    
    passed = 0
    for tc in test_cases:
        inputs = tc['inputs']
        expected = tc['expected']
        pred = decompiled(inputs)
        if pred == expected:
            passed += 1
    
    total = len(test_cases)
    pct = 100 * passed // total
    status = "PASS" if passed == total else "FAIL"
    print(f"{name}: {passed}/{total} ({pct}%) [{status}]")
    
    if passed != total:
        all_passed = False

print()
if all_passed:
    print("✓ All RNN tests pass with emitted code")
    sys.exit(0)
else:
    print("✗ Some RNN tests failed")
    sys.exit(1)
