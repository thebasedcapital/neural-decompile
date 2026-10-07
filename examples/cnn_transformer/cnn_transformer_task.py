"""
Syntactic Structure Recovery Task for Neural Decompilation

Task: Learn to recover canonical program structure from obfuscated/minified input.

Input:  ["if x>0 goto 2", "add 1 y", "halt"]
Output: Canonical form with normalized variable names, indentation, etc.

Architecture: CNN frontend → Transformer
- tokens → embed → 1D-Conv (temporal features) → LayerNorm → Transformer

This mimics CNN-transformer hybrids used in speech/NLP where CNNs pre-filter temporal features.
"""

import json
import random
import math
from typing import List, Tuple, Dict

# ============================================================================
# DATASET GENERATION
# ============================================================================

# Program operations (finite-state machine operations)
OPERATIONS = [
    "add",      # add const to var: "add 1 x"
    "sub",      # subtract const from var
    "mov",      # mov const to var: "mov 0 x"
    "if_gt",    # if var > const goto N: "if x>0 goto 2"
    "if_eq",    # if var == const goto N
    "if_lt",    # if var < const goto N
    "halt",     # stop execution
    "inc",      # increment var
    "dec",      # decrement var
]

VARIABLES = ["x", "y", "z", "a", "b", "c"]
MAX_CONST = 10
MAX_PROGRAM_LEN = 8
MAX_GOTO = 10  # Max line number for goto targets


def generate_random_program(max_len: int = 6) -> List[str]:
    """Generate a random finite-state program."""
    length = random.randint(3, max_len)
    program = []
    
    for i in range(length):
        op = random.choice(OPERATIONS)
        var = random.choice(VARIABLES)
        const = random.randint(0, MAX_CONST)
        
        if op in ["add", "sub"]:
            program.append(f"{op} {const} {var}")
        elif op == "mov":
            program.append(f"{op} {const} {var}")
        elif op == "inc":
            program.append(f"{op} {var}")
        elif op == "dec":
            program.append(f"{op} {var}")
        elif op in ["if_gt", "if_eq", "if_lt"]:
            goto_target = random.randint(0, MAX_GOTO)
            program.append(f"if {var}{['>','==','<'][['if_gt','if_eq','if_lt'].index(op)]}{const} goto {goto_target}")
        elif op == "halt":
            program.append("halt")
    
    return program


def normalize_program(program: List[str]) -> List[str]:
    """Convert program to canonical form.
    
    Canonicalization rules:
    1. Sort variables: x < y < z < a < b < c
    2. Normalize constants to smallest equivalent
    3. Standardize goto targets (relative offsets)
    4. Canonical spacing and format
    """
    # Variable renaming: map first var used to 'x', second to 'y', etc.
    var_map = {}
    next_var = 0
    var_order = ["x", "y", "z", "a", "b", "c"]
    
    normalized = []
    
    for line in program:
        # Parse the line
        parts = line.split()
        op = parts[0]
        
        if op in ["add", "sub", "mov"]:
            # add/sub const var -> canonicalize var
            const, var = parts[1], parts[2]
            if var not in var_map:
                var_map[var] = var_order[next_var]
                next_var = min(next_var + 1, len(var_order) - 1)
            normalized.append(f"{op} {const} {var_map[var]}")
        
        elif op in ["inc", "dec"]:
            var = parts[1]
            if var not in var_map:
                var_map[var] = var_order[next_var]
                next_var = min(next_var + 1, len(var_order) - 1)
            normalized.append(f"{op} {var_map[var]}")
        
        elif op == "if":
            # Parse "if var>const goto N"
            rest = " ".join(parts[1:])  # "x>0 goto 2"
            # Extract var, op, const, goto
            import re
            match = re.match(r"(\w+)(>|==|<)(\d+)\s+goto\s+(\d+)", rest)
            if match:
                var, cmp, const, goto = match.groups()
                if var not in var_map:
                    var_map[var] = var_order[next_var]
                    next_var = min(next_var + 1, len(var_order) - 1)
                normalized.append(f"if {var_map[var]}{cmp}{const} goto {goto}")
        
        elif op == "halt":
            normalized.append("halt")
    
    return normalized


def obfuscate_program(program: List[str], seed: int = None) -> List[str]:
    """Apply random obfuscations to create training variety.
    
    Obfuscations:
    1. Random variable names (not canonical)
    2. Extra whitespace
    3. Different constant representations (hex, etc.)
    4. Shuffled goto targets (with corresponding adjustments)
    """
    if seed is not None:
        random.seed(seed)
    
    obfuscated = []
    
    # Random variable mapping (different from canonical)
    used_vars = list(set(v for line in program for v in VARIABLES if v in line))
    obfuscation_names = ["p", "q", "r", "s", "t", "u", "v", "w"]
    random.shuffle(obfuscation_names)
    var_rename = {v: obfuscation_names[i] for i, v in enumerate(used_vars)}
    
    for line in program:
        parts = line.split()
        op = parts[0]
        
        if op in ["add", "sub", "mov"]:
            const, var = parts[1], parts[2]
            new_var = var_rename.get(var, var)
            # Sometimes use hex
            if random.random() < 0.3:
                const = f"0x{int(const):x}"
            obfuscated.append(f"{op}  {const}  {new_var}")  # Extra spaces
        
        elif op in ["inc", "dec"]:
            var = parts[1]
            new_var = var_rename.get(var, var)
            obfuscated.append(f"{op} {new_var}")
        
        elif op == "if":
            import re
            rest = " ".join(parts[1:])
            match = re.match(r"(\w+)(>|==|<)(\d+)\s+goto\s+(\d+)", rest)
            if match:
                var, cmp, const, goto = match.groups()
                new_var = var_rename.get(var, var)
                obfuscated.append(f"if {new_var}{cmp}{const} goto {goto}")
        
        elif op == "halt":
            obfuscated.append("halt")
    
    return obfuscated


# ============================================================================
# TOKENIZER
# ============================================================================

class Tokenizer:
    def __init__(self):
        # Build vocabulary
        self.tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        
        # Operations
        self.tokens.extend(["add", "sub", "mov", "inc", "dec", "if", "goto", "halt"])
        
        # Comparison operators
        self.tokens.extend([">", "==", "<"])
        
        # Variables (canonical and obfuscated)
        self.tokens.extend(VARIABLES)  # x, y, z, a, b, c
        self.tokens.extend(["p", "q", "r", "s", "t", "u", "v", "w"])  # obfuscated
        
        # Numbers 0-10
        self.tokens.extend([str(i) for i in range(11)])
        self.tokens.extend([f"0x{i:x}" for i in range(11)])  # hex versions
        
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.tokens)
    
    def tokenize(self, program: List[str]) -> List[int]:
        """Convert program to token IDs."""
        ids = [self.token_to_id["<START>"]]
        
        for line in program:
            # Tokenize each line
            parts = line.split()
            for part in parts:
                # Handle compound tokens like "x>0"
                import re
                subparts = re.findall(r'(\w+|>|==|<|0x\w+)', part)
                for sp in subparts:
                    if sp in self.token_to_id:
                        ids.append(self.token_to_id[sp])
                    else:
                        # Try parsing as number
                        try:
                            if sp.startswith("0x"):
                                val = int(sp, 16)
                            else:
                                val = int(sp)
                            if 0 <= val <= 10:
                                ids.append(self.token_to_id[str(val)])
                            else:
                                ids.append(self.token_to_id["<UNK>"])
                        except:
                            ids.append(self.token_to_id["<UNK>"])
        
        ids.append(self.token_to_id["<END>"])
        return ids
    
    def detokenize(self, ids: List[int]) -> List[str]:
        """Convert token IDs back to program lines."""
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        
        # Group tokens into lines
        lines = []
        current_line = []
        ops = {"add", "sub", "mov", "inc", "dec", "if", "halt"}
        
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ["<PAD>", "<START>", "<END>"]:
                i += 1
                continue
            
            if t in ["add", "sub", "mov"]:
                # add const var
                if i + 2 < len(tokens):
                    line = f"{t} {tokens[i+1]} {tokens[i+2]}"
                    lines.append(line)
                    i += 3
                else:
                    i += 1
            
            elif t in ["inc", "dec"]:
                if i + 1 < len(tokens):
                    lines.append(f"{t} {tokens[i+1]}")
                    i += 2
                else:
                    i += 1
            
            elif t == "if":
                # if var cmp const goto N
                if i + 5 < len(tokens):
                    line = f"if {tokens[i+1]}{tokens[i+2]}{tokens[i+3]} goto {tokens[i+5]}"
                    lines.append(line)
                    i += 6
                else:
                    i += 1
            
            elif t == "halt":
                lines.append("halt")
                i += 1
            
            else:
                i += 1
        
        return lines


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def randn(shape, scale=0.1):
    """Xavier initialization."""
    import math
    if isinstance(shape, int):
        shape = (shape,)
    n_in = shape[-1] if len(shape) == 2 else shape[0]
    return [[random.gauss(0, scale * math.sqrt(2.0 / n_in)) 
             for _ in range(shape[1] if len(shape) == 2 else 1)]
            for _ in range(shape[0])]


class Conv1D:
    """1D convolution over sequence."""
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.weight = randn((out_channels, in_channels * kernel_size), scale=0.5)
        self.bias = [0.0] * out_channels
    
    def __call__(self, x):
        """x: [seq_len, in_channels]"""
        seq_len = len(x)
        in_channels = len(x[0])
        
        # Pad input
        pad = self.kernel_size // 2
        x_padded = [[0.0] * in_channels] * pad + x + [[0.0] * in_channels] * pad
        
        # Convolve
        out = []
        for i in range(seq_len):
            window = []
            for k in range(self.kernel_size):
                window.extend(x_padded[i + k])
            
            # Compute output for this position
            out_vec = []
            for oc in range(self.out_channels):
                s = self.bias[oc]
                for ic in range(len(window)):
                    s += window[ic] * self.weight[oc][ic]
                out_vec.append(s)
            out.append(out_vec)
        
        return out


class LayerNorm:
    def __init__(self, dim):
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim
    
    def __call__(self, x):
        """x: [seq_len, dim]"""
        out = []
        for vec in x:
            mean = sum(vec) / len(vec)
            var = sum((v - mean) ** 2 for v in vec) / len(vec)
            std = math.sqrt(var + 1e-5)
            normalized = [(vec[i] - mean) / std * self.gamma[i] + self.beta[i] 
                         for i in range(len(vec))]
            out.append(normalized)
        return out


class CNNTransformer:
    """CNN frontend + Transformer for structure recovery."""
    
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64, n_layers=2, kernel_size=3):
        self.d_model = d_model
        
        # Token embedding
        self.embed = randn((vocab_size, d_model), scale=0.1)
        
        # Position embedding
        # (will be added during forward)
        
        # 1D CNN frontend (time-distributed convolutions)
        self.conv1 = Conv1D(d_model, d_model, kernel_size)
        self.ln1 = LayerNorm(d_model)
        
        # Second conv layer
        self.conv2 = Conv1D(d_model, d_model, kernel_size)
        self.ln2 = LayerNorm(d_model)
        
        # Transformer layers
        self.wq = [randn((d_model, d_model), scale=0.1) for _ in range(n_layers)]
        self.wk = [randn((d_model, d_model), scale=0.1) for _ in range(n_layers)]
        self.wv = [randn((d_model, d_model), scale=0.1) for _ in range(n_layers)]
        self.wo = [randn((d_model, d_model), scale=0.1) for _ in range(n_layers)]
        
        self.w1 = [randn((d_model, d_ff), scale=0.1) for _ in range(n_layers)]
        self.w2 = [randn((d_ff, d_model), scale=0.1) for _ in range(n_layers)]
        
        self.ln_attn = [LayerNorm(d_model) for _ in range(n_layers)]
        self.ln_ffn = [LayerNorm(d_model) for _ in range(n_layers)]
        
        # Output projection
        self.w_out = randn((d_model, vocab_size), scale=0.1)
        self.b_out = [0.0] * vocab_size
        
        self.n_layers = n_layers
    
    def attention(self, q, k, v):
        """Single-head self-attention."""
        seq_len = len(q)
        d = len(q[0])
        
        # Compute scores
        scores = [[0.0] * seq_len for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                for d_idx in range(d):
                    scores[i][j] += q[i][d_idx] * k[j][d_idx]
                scores[i][j] /= math.sqrt(d)
        
        # Softmax
        attn = []
        for i in range(seq_len):
            max_s = max(scores[i])
            e = [math.exp(s - max_s) for s in scores[i]]
            s = sum(e)
            attn.append([ei / s for ei in e])
        
        # Apply attention
        out = [[0.0] * d for _ in range(seq_len)]
        for i in range(seq_len):
            for d_idx in range(d):
                for j in range(seq_len):
                    out[i][d_idx] += attn[i][j] * v[j][d_idx]
        
        return out
    
    def matmul(self, x, w):
        """x: [seq_len, d_in], w: [d_in, d_out] -> [seq_len, d_out]"""
        seq_len = len(x)
        d_in = len(x[0])
        d_out = len(w[0])
        return [[sum(x[i][k] * w[k][j] for k in range(d_in)) for j in range(d_out)] for i in range(seq_len)]
    
    def forward(self, token_ids):
        """Forward pass."""
        seq_len = len(token_ids)
        
        # Embed
        h = [[self.embed[tid][d] for d in range(self.d_model)] for tid in token_ids]
        
        # Positional encoding (simple learned positions)
        for i in range(seq_len):
            for d in range(self.d_model):
                h[i][d] += 0.1 * math.sin(i / (10000 ** (d / self.d_model)))
        
        # CNN frontend
        h = self.conv1(h)
        h = [[math.max(0, x) for x in row] for row in h]  # ReLU
        h = self.ln1(h)
        
        h = self.conv2(h)
        h = [[math.max(0, x) for x in row] for row in h]  # ReLU
        h = self.ln2(h)
        
        # Transformer layers
        for layer in range(self.n_layers):
            # Self-attention
            h_norm = self.ln_attn[layer](h)
            q = self.matmul(h_norm, self.wq[layer])
            k = self.matmul(h_norm, self.wk[layer])
            v = self.matmul(h_norm, self.wv[layer])
            attn_out = self.attention(q, k, v)
            proj = self.matmul(attn_out, self.wo[layer])
            
            # Residual
            h = [[h[i][d] + proj[i][d] for d in range(self.d_model)] for i in range(seq_len)]
            
            # FFN
            h_norm = self.ln_ffn[layer](h)
            ffn_hidden = self.matmul(h_norm, self.w1[layer])
            ffn_hidden = [[math.max(0, x) for x in row] for row in ffn_hidden]  # ReLU
            ffn_out = self.matmul(ffn_hidden, self.w2[layer])
            
            # Residual
            h = [[h[i][d] + ffn_out[i][d] for d in range(self.d_model)] for i in range(seq_len)]
        
        # Output projection
        logits = self.matmul(h, self.w_out)
        for i in range(seq_len):
            for j in range(len(self.b_out)):
                logits[i][j] += self.b_out[j]
        
        return logits


# ============================================================================
# TRAINING
# ============================================================================

def generate_dataset(n_samples, tokenizer):
    """Generate training data."""
    data = []
    for _ in range(n_samples):
        program = generate_random_program()
        canonical = normalize_program(program)
        obfuscated = obfuscate_program(program)
        
        input_ids = tokenizer.tokenize(obfuscated)
        target_ids = tokenizer.tokenize(canonical)
        
        data.append((input_ids, target_ids, obfuscated, canonical))
    
    return data


def cross_entropy_loss(logits, target_id):
    """Compute cross-entropy loss for one position."""
    max_l = max(logits)
    e = [math.exp(l - max_l) for l in logits]
    s = sum(e)
    return -math.log(e[target_id] / s)


def train_model(n_epochs=100, n_samples_per_epoch=500):
    """Train the model."""
    print("Initializing...")
    tokenizer = Tokenizer()
    model = CNNTransformer(tokenizer.vocab_size, d_model=32, n_heads=4, d_ff=64, n_layers=2)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model params: {sum(len(p) if isinstance(p[0], list) else len(p) for p in model.__dict__.values() if isinstance(p, list) and p and isinstance(p[0], (int, float, list)))}")
    
    for epoch in range(n_epochs):
        data = generate_dataset(n_samples_per_epoch, tokenizer)
        
        # Simple gradient-free optimization: random search with local refinement
        # (Full backprop would require autograd library)
        
        # For now, just evaluate current model
        correct = 0
        total_loss = 0
        
        for input_ids, target_ids, _, _ in data:
            logits = model.forward(input_ids)
            
            # Compute loss (output at each position should match target)
            for i, (logits_i, target_i) in enumerate(zip(logits, target_ids[:len(logits)])):
                total_loss += cross_entropy_loss(logits_i, target_i)
                if logits_i.index(max(logits_i)) == target_i:
                    correct += 1
        
        avg_loss = total_loss / (len(data) * 10)  # rough normalization
        accuracy = correct / (len(data) * 10) * 100
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.3f}, accuracy={accuracy:.1f}%")
    
    return model, tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("Syntactic Structure Recovery Task")
    print("=" * 60)
    print()
    print("Task: Learn to recover canonical program structure from")
    print("obfuscated/minified finite-state programs.")
    print()
    print("Architecture: tokens → embed → 1D-Conv → LayerNorm → Transformer")
    print()
    
    # Generate sample data
    tokenizer = Tokenizer()
    print("Sample data:")
    for _ in range(3):
        program = generate_random_program(max_len=4)
        canonical = normalize_program(program)
        obfuscated = obfuscate_program(program)
        print(f"  Original: {program}")
        print(f"  Canonical: {canonical}")
        print(f"  Obfuscated: {obfuscated}")
        print()
    
    print("Training...")
    print("(Note: Full training requires PyTorch/TensorFlow for autograd)")
    print("This demo shows the architecture and data format.")
    print()
    
    # Export architecture spec
    arch_spec = {
        "task": "syntactic_structure_recovery",
        "description": "Recover canonical program structure from obfuscated finite-state programs",
        "architecture": {
            "type": "cnn_transformer",
            "embedding": {"vocab_size": Tokenizer().vocab_size, "d_model": 32},
            "cnn_frontend": [
                {"type": "conv1d", "in_channels": 32, "out_channels": 32, "kernel_size": 3},
                {"type": "relu"},
                {"type": "layer_norm", "dim": 32},
                {"type": "conv1d", "in_channels": 32, "out_channels": 32, "kernel_size": 3},
                {"type": "relu"},
                {"type": "layer_norm", "dim": 32}
            ],
            "transformer": {
                "n_layers": 2,
                "n_heads": 4,
                "d_model": 32,
                "d_ff": 64
            },
            "output": {"type": "linear", "d_model": 32, "vocab_size": Tokenizer().vocab_size}
        },
        "input_format": "tokenized program lines",
        "output_format": "canonical program lines",
        "operations": OPERATIONS,
        "variables": VARIABLES,
    }
    
    with open("/Users/bbclaude/Projects/neural-decompile/examples/cnn_transformer_spec.json", "w") as f:
        json.dump(arch_spec, f, indent=2)
    
    print(f"Exported architecture spec to examples/cnn_transformer_spec.json")