# Exact finite-state implementation of the certified integer model.
# Input: binary symbols (0 or 1), not one-hot vectors.
TRANSITIONS = [[0, 1], [2, 0], [1, 3], [4, 2], [5, 0], [6, 2], [1, 0]]
OUTPUTS = [1, 0, 0, 0, 0, 0, 0]

def decompiled(bits):
    state = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError('expected binary symbols')
        state = TRANSITIONS[state][bit]
    return OUTPUTS[state]
