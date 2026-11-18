"""Check the hand-written backprop against numerical gradients: (f(w+h) - f(w-h)) / 2h."""
import numpy as np
from model import MLP, make_dataset

names = ["emma", "olivia", "ava", "noah", "liam"]
X, Y = make_dataset(names, 3)
m = MLP(hidden=16, emb=4, seed=1)
m.W2 = np.random.default_rng(2).normal(0, 0.5, m.W2.shape)      # non-trivial weights so gradients aren't tiny
m.loss(X, Y); g = m.backward(Y)
worst = 0
rng = np.random.default_rng(0)
for name, p in m.params().items():
    for _ in range(8):
        i = tuple(rng.integers(s) for s in p.shape)
        old = p[i]
        p[i] = old + 1e-5; lp = m.loss(X, Y)
        p[i] = old - 1e-5; lm = m.loss(X, Y)
        p[i] = old
        num = (lp - lm) / 2e-5
        rel = abs(num - g[name][i]) / max(1e-8, abs(num) + abs(g[name][i]))
        worst = max(worst, rel)
print(f"max relative error between backprop and numerical gradient: {worst:.2e}")
assert worst < 1e-5, "backprop is wrong"
print("gradient check passed ✓")
