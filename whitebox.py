"""WhiteBox v0: distill a trained black box into a Walsh-Fourier hypergraph.

    python whitebox.py        (numpy only, CPU, under a minute)

A black box f: {-1,+1}^n -> {-1,+1} has exactly one expansion
    f(x) = sum_S theta_S * prod_{i in S} x_i,   theta_S = E[f(x) * chi_S(x)]
Each S with theta_S != 0 is a branch (hyperedge). Parseval: sum_S theta_S^2 = 1,
so the share of the model a graph H explains is fidelity(H) = sum_{S in H} theta_S^2.

Two read-outs:
  exact  - fast Walsh-Hadamard transform over the whole cube (small n only)
  query  - Kushilevitz-Mansour / Goldreich-Levin bucket descent, random queries only,
           cost poly(n, 1/tau) instead of 2^n or C(n, k)
"""
import json
import os
import time

import numpy as np

SIGMA = 0.35


# ------------------------------------------------------------ toy world (BindingClaude T2/T3)
def world(task, D, n, rng):
    """Noisy +-1 evidence. T2: y = s*e1*e2 (3-way binding). T3: m = s*K is keyed, y = m*K*e1*e2."""
    b = lambda *shape: rng.choice([-1., 1.], size=shape)
    e1, e2, s, K, clut = b(n), b(n), b(n), b(n), b(n, D)
    if task == "T2":
        cols, X = ["e1", "e2", "m"], [e1, e2, s]
    else:
        cols, X = ["e1", "e2", "m", "K"], [e1, e2, s * K, K]
    cols += [f"c{i}" for i in range(D)]
    X = np.column_stack(X + [clut])
    return X + rng.normal(0, SIGMA, X.shape), s * e1 * e2, cols


# ------------------------------------------------------------ the black box: an ordinary MLP
def train_mlp(Z, y, seed=0, hidden=64, steps=1500, lr=0.02):
    r = np.random.default_rng(seed)
    d = Z.shape[1]
    P = [r.normal(0, d ** -.5, (d, hidden)), np.zeros(hidden), r.normal(0, hidden ** -.5, hidden), np.zeros(1)]
    M = [np.zeros_like(p) for p in P]
    V = [np.zeros_like(p) for p in P]
    t01 = (y + 1) / 2
    for t in range(1, steps + 1):
        h = np.tanh(Z @ P[0] + P[1])
        g = (1 / (1 + np.exp(-(h @ P[2] + P[3]))) - t01) / len(y)
        gh = np.outer(g, P[2]) * (1 - h ** 2)
        G = [Z.T @ gh, gh.sum(0), h.T @ g, np.array([g.sum()])]
        for i in range(4):
            M[i] = .9 * M[i] + .1 * G[i]
            V[i] = .999 * V[i] + .001 * G[i] ** 2
            P[i] -= lr * (M[i] / (1 - .9 ** t)) / (np.sqrt(V[i] / (1 - .999 ** t)) + 1e-8)
    return lambda X: np.where(np.tanh(X @ P[0] + P[1]) @ P[2] + P[3] >= 0, 1., -1.)


# ------------------------------------------------------------ read-out 1: exact spectrum
def walsh_exact(f, n):
    X = 1. - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)   # bit i of index = coord i
    a, h = f(X), 1
    while h < len(a):
        a = a.reshape(-1, 2 * h)
        a = np.concatenate([a[:, :h] + a[:, h:], a[:, :h] - a[:, h:]], 1).ravel()
        h *= 2
    return a / len(a)                                                  # theta indexed by bitmask


# ------------------------------------------------------------ read-out 2: queries only
def walsh_km(f, n, rng, tau=0.3, q=600):
    """Kushilevitz-Mansour. Bucket (alpha, k) = all S whose first-k part is alpha.
    Its mass sum theta_S^2 = E[f(x)chi_a(x) f(y)chi_a(y)], x and y sharing coords >= k."""
    live, queries = [[]], 0
    for k in range(1, n + 1):
        nxt = []
        for a in live:
            for b in (a, a + [k - 1]):
                x = rng.choice([-1., 1.], (q, n))
                y = x.copy()
                y[:, :k] = rng.choice([-1., 1.], (q, k))
                ch = (lambda v: np.prod(v[:, b], 1)) if b else (lambda v: 1.)
                queries += 2 * q
                if np.mean(f(x) * ch(x) * f(y) * ch(y)) > tau ** 2 / 2:
                    nxt.append(b)
        live = nxt
    # estimate each surviving coefficient
    x = rng.choice([-1., 1.], (4000, n))
    fx = f(x)
    queries += 4000
    est = {tuple(S): float(np.mean(fx * np.prod(x[:, S], 1))) for S in live}
    return {S: v for S, v in est.items() if abs(v) >= tau / 2}, queries   # final coefficient threshold


# ------------------------------------------------------------ run
def main():
    os.makedirs("results", exist_ok=True)
    out, lines = [], ["| black box | n | top branch (exact) | theta | fidelity | degree profile | graph agrees with MLP | task acc MLP / graph | KM found | KM queries | 2^n |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
    for task, D in (("T2", 8), ("T3", 8), ("T2", 16)):
        rng = np.random.default_rng(1)
        Z, y, cols = world(task, D, 2048, rng)
        n = len(cols)
        f = train_mlp(Z, y)
        th = walsh_exact(f, n)
        top = int(np.argmax(th ** 2))
        S = [i for i in range(n) if top >> i & 1]
        deg = np.array([bin(m).count("1") for m in range(2 ** n)])
        W = {k: round(float((th[deg == k] ** 2).sum()), 4) for k in range(n + 1) if (th[deg == k] ** 2).sum() > 1e-3}
        Zt, yt, _ = world(task, D, 4000, np.random.default_rng(9))
        graph = np.sign(th[top]) * np.prod(np.sign(Zt[:, S]), 1)       # the one-branch white box
        t0 = time.time()
        km, nq = walsh_km(f, n, np.random.default_rng(2))
        r = dict(task=task, n=n, branch=[cols[i] for i in S], theta=float(th[top]),
                 fidelity=float(th[top] ** 2), degree_profile=W,
                 graph_vs_mlp=float((graph == f(Zt)).mean()),
                 mlp_acc=float((f(Zt) == yt).mean()), graph_acc=float((graph == yt).mean()),
                 km={"+".join(cols[i] for i in k): round(v, 3) for k, v in km.items()},
                 km_queries=nq, km_seconds=round(time.time() - t0, 2))
        out.append(r)
        lines.append(f"| MLP on {task} | {n} | {{{', '.join(r['branch'])}}} | {r['theta']:.3f} | {r['fidelity']:.3f} | {W} | "
                     f"{r['graph_vs_mlp']:.3f} | {r['mlp_acc']:.3f} / {r['graph_acc']:.3f} | {list(r['km'])} | {nq:,} | {2 ** n:,} |")
        print(lines[-1], flush=True)
    json.dump(out, open("results/results.json", "w"), indent=2)
    open("results/LEDGER.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
