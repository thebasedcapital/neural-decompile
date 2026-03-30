#!/usr/bin/env python3
"""Baseline evaluation of neural-decompile on all existing examples."""
import subprocess
import json
import os

def run_eval():
    examples_dir = 'examples'
    results = []
    
    # Find all weight files (not test files)
    weight_files = []
    for root, dirs, files in os.walk(examples_dir):
        for f in files:
            if f.endswith('.json') and 'test' not in f.lower():
                weight_files.append(os.path.join(root, f))
    
    print("=" * 70)
    print("BASELINE EVAL: Neural Decompiler on All Examples")
    print("=" * 70)
    
    for weights_file in sorted(weight_files):
        # Find corresponding test file
        base = weights_file.replace('.json', '')
        test_file = base + '_tests.json'
        if not os.path.exists(test_file):
            # Try regex/ subdirectory
            parts = weights_file.split('/')
            if len(parts) > 2:
                test_file = '/'.join(parts[:-1]) + '/' + parts[-1].replace('.json', '_tests.json')
        
        if not os.path.exists(test_file):
            print(f"\n{weights_file}: NO TEST FILE FOUND")
            continue
        
        # Run verification
        result = subprocess.run(
            ['./target/release/nd', 'verify', weights_file, test_file],
            capture_output=True, text=True
        )
        
        # Parse result
        output = result.stdout.strip()
        passed = 0
        total = 0
        is_perfect = False
        
        if 'passed' in output:
            try:
                # Extract "X/Y passed"
                parts = output.split('/')
                if len(parts) >= 2:
                    passed = int(parts[0].split()[-1])
                    total = int(parts[1].split()[0])
                    is_perfect = 'PERFECT' in output or passed == total
            except:
                pass
        
        # Run decompile to check output quality
        decomp_result = subprocess.run(
            ['./target/release/nd', 'decompile', weights_file],
            capture_output=True, text=True
        )
        
        decomp_output = decomp_result.stdout
        has_code = 'def ' in decomp_output or 'pub fn' in decomp_output
        has_integer_pct = 'integer' in decomp_result.stderr.lower()
        
        # Determine model type
        with open(weights_file) as f:
            data = json.load(f)
        model_type = 'Transformer' if 'd_model' in data else 'RNN'
        size_info = ''
        if model_type == 'RNN':
            size_info = f"hidden={data.get('hidden_dim', '?')}"
        else:
            size_info = f"d_model={data.get('d_model', '?')}, layers={data.get('n_layers', 1)}"
        
        # Score
        fidelity_ok = is_perfect
        interpretable = has_code
        complete = has_integer_pct
        
        score = sum([fidelity_ok, interpretable, complete])
        max_score = 3
        
        status = "✓" if is_perfect else "✗"
        print(f"\n{status} {weights_file}")
        print(f"   Type: {model_type} ({size_info})")
        print(f"   Verification: {passed}/{total} {'✓ PERFECT' if is_perfect else ''}")
        print(f"   Decompiled: {'YES' if has_code else 'NO'}, Stats: {'YES' if has_integer_pct else 'NO'}")
        print(f"   Score: {score}/{max_score}")
        
        results.append({
            'file': weights_file,
            'type': model_type,
            'passed': passed,
            'total': total,
            'perfect': is_perfect,
            'score': score,
            'max_score': max_score
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_score = sum(r['score'] for r in results)
    max_total = sum(r['max_score'] for r in results)
    perfect_count = sum(1 for r in results if r['perfect'])
    
    print(f"Models tested: {len(results)}")
    print(f"Perfect verification: {perfect_count}/{len(results)}")
    print(f"Total score: {total_score}/{max_total} ({100*total_score/max_total:.1f}%)")
    
    rnn_results = [r for r in results if r['type'] == 'RNN']
    transformer_results = [r for r in results if r['type'] == 'Transformer']
    
    if rnn_results:
        rnn_perfect = sum(1 for r in rnn_results if r['perfect'])
        print(f"RNNs: {rnn_perfect}/{len(rnn_results)} perfect")
    
    if transformer_results:
        tf_perfect = sum(1 for r in transformer_results if r['perfect'])
        print(f"Transformers: {tf_perfect}/{len(transformer_results)} perfect")
    
    # Gap analysis
    print("\n" + "=" * 70)
    print("GAP ANALYSIS (What's Missing for Journal-Worthy)")
    print("=" * 70)
    
    gaps = []
    if perfect_count < len(results):
        gaps.append("✗ Not all models verify at 100%")
    if len(transformer_results) < 3:
        gaps.append("✗ Only {} transformers — need more variety".format(len(transformer_results)))
    if all(r['perfect'] for r in results) and len(results) < 10:
        gaps.append("✗ Too few examples — need 10+ diverse models")
    
    # Check for novelty
    gaps.append("✗ No novel algorithm discovery — all examples are known algorithms (parity, mod, regex)")
    gaps.append("✗ No large models — all are tiny (<10 neurons/heads)")
    gaps.append("✗ No formal proofs in main repo — Kani proofs exist but not integrated")
    gaps.append("✗ No real-world task — all are toy math/regex tasks")
    
    for gap in gaps:
        print(gap)
    
    return results, gaps

if __name__ == '__main__':
    results, gaps = run_eval()
