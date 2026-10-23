# WhiteBox — distilling a black box into a hypergraph

**v0, a beginning.** A trained network is read back out as an explicit hypergraph of branches,
with a number that says how much of the network the graph explains. numpy only, CPU, under a
minute: `python whitebox.py`. Raw output in [results/](results/).

## The idea

Any model on binary variables, f: {−1,+1}ⁿ → {−1,+1}, has exactly one Walsh–Fourier expansion

  f(x) = Σ_S θ_S · χ_S(x),  χ_S(x) = Π_{i∈S} xᵢ,  θ_S = E[f(x)·χ_S(x)]

- **Branches.** Each set S with θ_S ≠ 0 is a hyperedge, a group of pieces that act together. A
  binding is a large θ_S with |S| ≥ 3. This is the energy E = −Σ_b J_b χ_b from BindingClaude
  read backwards: J_b = θ_b.
- **Fidelity (Parseval).** For a ±1 model Σ_S θ_S² = 1, so a graph H explains exactly
  fidelity(H) = Σ_{S∈H} θ_S² of the model. 1 − fidelity is the part still black.
- **Degree profile.** W^k = Σ_{|S|=k} θ_S² says how much of the model lives in single pieces, in
  pairs, and in true bindings.
- **A model to query removes the search explosion.** Growing branches blindly costs about C(N, k)
  (BindingClaude Part 2). With the model as an oracle, Kushilevitz–Mansour / Goldreich–Levin
  splits sets by their first j coordinates and keeps only buckets whose mass
  E[f(x)χ_α(x)·f(y)χ_α(y)] exceeds τ²/2 (x and y sharing the remaining coordinates). Every
  coefficient above τ is found in poly(n, 1/τ) queries, whatever the binding order.
- **With open weights** you can also query any hidden unit, and the discrete derivative
  Dᵢf(x) = (f(x^{i→+1}) − f(x^{i→−1}))/2 satisfies E[D_S f] = θ_S exactly. The graph can be read
  directly from the weights only for polynomial networks (multiplying branches, bilinear or gated
  layers); for ReLU or softmax it has to come from queries.

## First result

The black boxes are ordinary MLPs (64 tanh units, Adam), trained on noisy evidence (σ = 0.35)
from BindingClaude's worlds. T2: y = s·e1·e2, a 3-way binding. T3: the message is keyed,
m = s·K, so y = m·K·e1·e2, a 4-way binding. Plus 8 or 16 clutter pieces. The MLP is given no
branches.

| black box | n | exact top branch | fidelity | degree profile | graph agrees with MLP | task acc: MLP / graph | query read-out | queries | 2ⁿ |
|---|---|---|---|---|---|---|---|---|---|
| MLP on T2 | 11 | {e1, e2, m} | 1.000 | W³ = 1 | 98.7% | 98.3% / 99.3% | {e1, e2, m} | 32,800 | 2,048 |
| MLP on T3 | 12 | {e1, e2, m, K} | 1.000 | W⁴ = 1 | 97.3% | 97.0% / 99.1% | {e1, e2, m, K} | 37,600 | 4,096 |
| MLP on T2 | 19 | {e1, e2, m} | 1.000 | W³ = 1 | 98.3% | 97.9% / 99.1% | {e1, e2, m} | 56,800 | 524,288 |

- On clean inputs each trained network **is** exactly one branch: all of its spectrum sits on the
  true binding.
- Run as a one-branch white box, the graph does slightly better on the noisy task than the MLP it
  came from. The rule is applied cleanly instead of being smeared over 64 units.
- The query read-out finds the same branch without enumerating the input space. On T3 it also
  passed one spurious bucket ({e1, e2, m, K, c7}), whose estimated coefficient is −0.003, so a
  final coefficient threshold removes it.

## What this does not show

- These are toys where the answer is a single clean parity. A one-to-one recovery shows the
  instrument works, not what a real model's graph looks like.
- **The variables decide everything.** On raw pixels or tokens a model is not sparse. The graph is
  small only if the ±1 variables are the right pieces (a token or segment present/absent, a
  feature switched on or off, a Sihti band's sign). Choose the wrong basis and the bindings look
  like noise (Välikäsi's lesson).
- Real models are probably mostly low-degree (large W¹, W²) with a thin high-order tail. Fidelity
  says how much is left unexplained.
- Query count grows linearly in n here, but with q = 600 fixed per bucket; the noise in bucket
  estimates is what produced the spurious T3 bucket.

## Next

1. Run the distilled graph as a machine (BindingClaude Part 3 settling) and check every edit
   against the original model: delete branch #12 from the graph, make the matching change to the
   model's input, and see whether the model changes as predicted. Each knockout becomes a
   falsifiable claim about the black box.
2. First real target: Välikäsi. A's B-context cues as the toggles, the note feature (or C's
   answer) as the output, the 0.5B model as the oracle. That is the keyed reader, and it would show
   whether the leak is a high-order term invisible to linear probes.

## Prior art

Walsh–Fourier analysis of Boolean functions (O'Donnell, *Analysis of Boolean Functions*);
Goldreich & Levin 1989; Kushilevitz & Mansour 1993; Linial, Mansour & Nisan 1993 (low-degree
concentration); Pearce et al. 2024 (bilinear MLPs read from weights); Kang et al. 2025, SPEX
(sparse Fourier interactions of LLM tokens from masked queries). Lineage: SighImageSuper → Sihti
→ ClaudeRepo1 → Välikäsi → BindingClaude → WhiteBox.
