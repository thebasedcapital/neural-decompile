# Exact finite-state implementation of the certified integer model.
# Input: binary symbols (0 or 1), not one-hot vectors.
TRANSITIONS = [[0, 1], [0, 2], [2, 2]]
OUTPUTS = [0, 0, 1]

def decompiled(bits):
    state = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError('expected binary symbols')
        state = TRANSITIONS[state][bit]
    return OUTPUTS[state]
