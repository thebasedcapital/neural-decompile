#!/usr/bin/env python3
"""
Train an RNN on binary addition with carry.
Task: Given two n-bit binary numbers (LSB first), output their sum (n+1 bits).
Example: 1011 + 0110 = 10001 (11 + 6 = 17)
This requires learning the carry mechanism — a non-trivial sequential algorithm.
"""

import json
import random
import numpy as np

def generate_addition_data(n_bits, n_samples):
    """Generate binary addition examples."""
    X = []  # Input sequences: each step is [bit_a, bit_b]
    y = []  # Expected outputs: sum bit at each step
    
    for _ in range(n_samples):
        # Generate two random n-bit numbers
        a = [random.randint(0, 1) for _ in range(n_bits)]
        b = [random.randint(0, 1) for _ in range(n_bits)]
        
        # Compute sum with carry
        carry = 0
        sum_bits = []
        for i in range(n_bits):
            total = a[i] + b[i] + carry
            sum_bits.append(total % 2)
            carry = total // 2
        sum_bits.append(carry)  # Final carry bit
        
        # Input: sequence of [a_i, b_i] pairs
        X.append([[a[i], b[i]] for i in range(n_bits)])
        y.append(sum_bits)
    
    return X, y

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class SimpleRNN:
    """Simple RNN with one hidden layer."""
    
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Xavier init
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.W_hx = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (hidden_dim + input_dim))
        self.b_h = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (output_dim + hidden_dim))
        self.b_y = np.zeros(output_dim)
    
    def forward(self, sequence):
        """Forward pass through sequence."""
        h = np.zeros(self.hidden_dim)
        outputs = []
        
        for x in sequence:
            x = np.array(x)
            h = relu(np.dot(self.W_hh, h) + np.dot(self.W_hx, x) + self.b_h)
            logits = np.dot(self.W_y, h) + self.b_y
            outputs.append(logits)
        
        return outputs, h
    
    def train(self, X, y, epochs=500, lr=0.01, l1_lambda=0.01):
        """Train with L1 regularization to encourage integer weights."""
        n = len(X)
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Reset gradients
            dW_hh = np.zeros_like(self.W_hh)
            dW_hx = np.zeros_like(self.W_hx)
            db_h = np.zeros_like(self.b_h)
            dW_y = np.zeros_like(self.W_y)
            db_y = np.zeros_like(self.b_y)
            
            for i in range(n):
                seq = X[i]
                target = y[i]
                
                # Forward
                h = np.zeros(self.hidden_dim)
                h_states = [h.copy()]
                pre_activations = []
                
                for x in seq:
                    x = np.array(x)
                    pre_h = np.dot(self.W_hh, h) + np.dot(self.W_hx, x) + self.b_h
                    pre_activations.append(pre_h)
                    h = relu(pre_h)
                    h_states.append(h.copy())
                
                # Backward through time
                dh_next = np.zeros(self.hidden_dim)
                
                for t in reversed(range(len(seq))):
                    # Output gradient (MSE loss)
                    logits = np.dot(self.W_y, h_states[t+1]) + self.b_y
                    dlogits = 2 * (logits - np.eye(self.output_dim)[target[t]] if target[t] < self.output_dim else np.zeros(self.output_dim))
                    
                    # For binary output, we just want logit[1] > logit[0] for 1, vice versa
                    # Simplified: treat as regression to [0, 1]
                    target_onehot = np.zeros(self.output_dim)
                    if target[t] < self.output_dim:
                        target_onehot[target[t]] = 1
                    dlogits = 2 * (logits - target_onehot)
                    
                    dW_y += np.outer(dlogits, h_states[t+1])
                    db_y += dlogits
                    
                    dh = np.dot(self.W_y.T, dlogits) + dh_next
                    dpre = dh * relu_deriv(pre_activations[t])
                    
                    dW_hh += np.outer(dpre, h_states[t])
                    dW_hx += np.outer(dpre, np.array(seq[t]))
                    db_h += dpre
                    
                    dh_next = np.dot(self.W_hh.T, dpre)
                
                total_loss += sum((logits - target_onehot) ** 2)
            
            # Add L1 regularization gradient
            dW_hh += l1_lambda * np.sign(self.W_hh)
            dW_hx += l1_lambda * np.sign(self.W_hx)
            db_h += l1_lambda * np.sign(self.b_h)
            dW_y += l1_lambda * np.sign(self.W_y)
            db_y += l1_lambda * np.sign(self.b_y)
            
            # Update
            self.W_hh -= lr * dW_hh / n
            self.W_hx -= lr * dW_hx / n
            self.b_h -= lr * db_h / n
            self.W_y -= lr * dW_y / n
            self.b_y -= lr * db_y / n
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / n:.4f}")
        
        print(f"Final Loss: {total_loss / n:.4f}")
    
    def to_dict(self):
        """Export weights in neural-decompile format."""
        return {
            "W_hh": self.W_hh.tolist(),
            "W_hx": self.W_hx.tolist(),
            "b_h": self.b_h.tolist(),
            "W_y": self.W_y.tolist(),
            "b_y": self.b_y.tolist(),
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

def main():
    print("Training RNN on binary addition with carry...")
    print("Task: Given two n-bit binary numbers (LSB first), output their sum")
    print()
    
    # Generate data
    n_bits = 4
    n_train = 200
    n_test = 50
    
    X_train, y_train = generate_addition_data(n_bits, n_train)
    X_test, y_test = generate_addition_data(n_bits, n_test)
    
    print(f"Training on {n_train} examples of {n_bits}-bit addition")
    print(f"Testing on {n_test} examples")
    print()
    
    # Train
    rnn = SimpleRNN(input_dim=2, hidden_dim=50, output_dim=2)
    rnn.train(X_train, y_train, epochs=500, lr=0.005, l1_lambda=0.02)
    
    # Evaluate
    correct = 0
    for i in range(len(X_test)):
        outputs, _ = rnn.forward(X_test[i])
        pred = [1 if o[1] > o[0] else 0 for o in outputs]
        # Pad pred to match target length
        while len(pred) < len(y_test[i]):
            pred.append(0)
        pred = pred[:len(y_test[i])]
        
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test)
    print(f"\nTest accuracy: {accuracy*100:.1f}% ({correct}/{len(X_test)})")
    
    if accuracy < 0.8:
        print("WARNING: Low accuracy — model may not learn the algorithm cleanly")
    
    # Save weights
    weights = rnn.to_dict()
    with open("examples/binary_add.json", "w") as f:
        json.dump(weights, f, indent=2)
    
    print(f"\nSaved weights to examples/binary_add.json")
    
    # Generate test file
    test_cases = []
    for i in range(min(20, len(X_test))):
        test_cases.append({
            "inputs": X_test[i],
            "expected": y_test[i][-1] if y_test[i][-1] < 2 else 1  # Simplified: just predict last bit
        })
    
    # Actually, for full sequence prediction we need a different test format
    # For now, just save the weights
    print("Note: Test file generation needs sequence-to-sequence format")

if __name__ == "__main__":
    main()
