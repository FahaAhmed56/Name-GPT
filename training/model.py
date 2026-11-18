"""Two character-level language models for names.

* Bigram:  P(next | previous char), counted from data (the classic baseline).
* MLP:     Bengio et al. (2003): embed the previous 3 chars, a tanh hidden layer, softmax over 27 chars.
           Forward and backward passes written by hand in NumPy.

'.' marks the start and end of a name, so the vocabulary is '.' + a..z = 27 tokens.
"""
import numpy as np

CHARS = "." + "abcdefghijklmnopqrstuvwxyz"
STOI = {c: i for i, c in enumerate(CHARS)}
V = len(CHARS)


def make_dataset(names, block):
    """Every position in every name becomes one example: (previous `block` chars) -> next char."""
    X, Y = [], []
    for n in names:
        ctx = [0] * block
        for ch in n + ".":
            ix = STOI[ch]
            X.append(ctx); Y.append(ix)
            ctx = ctx[1:] + [ix]
    return np.array(X), np.array(Y)


class Bigram:
    def fit(self, names, smoothing=1):
        counts = np.full((V, V), smoothing, float)          # +1 smoothing: no probability is ever exactly 0
        for n in names:
            seq = [0] + [STOI[c] for c in n] + [0]
            for a, b in zip(seq, seq[1:]):
                counts[a, b] += 1
        self.P = counts / counts.sum(1, keepdims=True)
        return self

    def loss(self, X, Y):                                   # X: (n, block); only the last char matters
        return float(-np.log(self.P[X[:, -1], Y]).mean())


class MLP:
    def __init__(self, block=3, emb=10, hidden=200, seed=42):
        rng = np.random.default_rng(seed)
        self.block, self.emb = block, emb
        self.C = rng.normal(0, 1, (V, emb))                                    # character embeddings
        self.W1 = rng.normal(0, 1, (block * emb, hidden)) * (5 / 3) / np.sqrt(block * emb)   # Kaiming init for tanh
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1, (hidden, V)) * 0.01                         # small: start with near-uniform guesses
        self.b2 = np.zeros(V)

    def params(self):
        return {"C": self.C, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        self.X = X
        self.e = self.C[X].reshape(len(X), -1)              # (n, block*emb): look up + concatenate embeddings
        self.h = np.tanh(self.e @ self.W1 + self.b1)        # (n, hidden)
        logits = self.h @ self.W2 + self.b2                 # (n, 27)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        self.probs = p / p.sum(1, keepdims=True)
        return self.probs

    def loss(self, X, Y):
        p = self.forward(X)
        return float(-np.log(p[np.arange(len(Y)), Y] + 1e-12).mean())

    def backward(self, Y):
        """Backpropagation by hand: the chain rule, from the loss back to every weight."""
        n = len(Y)
        dlogits = self.probs.copy(); dlogits[np.arange(n), Y] -= 1; dlogits /= n     # softmax + cross-entropy
        g = {"W2": self.h.T @ dlogits, "b2": dlogits.sum(0)}
        dh = dlogits @ self.W2.T
        dpre = dh * (1 - self.h ** 2)                        # tanh'(x) = 1 - tanh(x)^2
        g["W1"] = self.e.T @ dpre; g["b1"] = dpre.sum(0)
        de = (dpre @ self.W1.T).reshape(n, self.block, self.emb)
        dC = np.zeros_like(self.C)
        np.add.at(dC, self.X, de)                            # scatter gradients back to the embedding rows used
        g["C"] = dC
        return g

    def step(self, g, lr):
        for k, p in self.params().items():
            p -= lr * g[k]

    def export(self):
        return {k: np.round(v, 6).tolist() for k, v in self.params().items()} | {"block": self.block}
