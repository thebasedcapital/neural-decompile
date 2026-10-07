# Auto-decompiled by neural-decompile
# 100% of weights are exact integers
# Hidden dim: 2, Input dim: 2, Output dim: 2

def decompiled(input_sequence):
    """Execute the quantized ReLU recurrent network."""
    h = [0.0] * 2
    for x in input_sequence:
        h0 = 0
        h1 = max(0, 2*h[1] + -1*x[0] + 2*x[1] + -1)
        h = [h0, h1]
    logits = []
    logits.append(-2*h[1] + 3)
    logits.append(2*h[1] + -3)
    return logits.index(max(logits))
