# Why theory isn't speed

*Part 4, the last part of the guided tour. Builds on
[part 3](03-how-bmssp-breaks-it.md): BMSSP computes exact distances in
`O(m log^(2/3) n)`, asymptotically beating Dijkstra's `O(m + n log n)`.
Now we race them.*

## What big-O actually promises

`O(m log^(2/3) n)` is a statement about the *shape* of the cost curve as `n`
grows toward infinity. It deliberately ignores constant factors: an algorithm
doing `1000 · m log^(2/3) n` operations and one doing `2 · m log^(2/3) n`
have the same big-O. For theory that's the right abstraction — constants
depend on implementation details, machines, compilers. For a stopwatch,
constants are the whole game at any size you can actually run.

So the honest comparison is: Dijkstra costs about `c₁ · (m + n log n)` with a
tiny `c₁`, BMSSP costs about `c₂ · m log^(2/3) n` with some unknown `c₂`, and
the asymptotic advantage only cashes out when `log^(1/3) n` outgrows
`c₂/c₁`. Everything hinges on how big `c₂` is. This repository was built to
measure it.

## Where the constant comes from

Three sources, all visible in part 3's construction if you squint:

**The degree-capping transform.** The paper's analysis needs every vertex to
have few edges, so the graph is first rewritten: high-degree vertices become
little cycles of stand-ins. Measured on this repository's benchmark graph
(4 million edges), the rewrite hands the algorithm 8 million vertices and 12
million edges to process — the problem got about 3× bigger before any actual
work began.

**Bookkeeping dwarfing the work.** At realistic sizes the parameters
degenerate: `log₂ n ≈ 20` gives `k = ⌊20^(1/3)⌋ = 2`, so the "big" recursive
machinery — pivot finding, the block queue, recursion frames — fires millions
of times on subproblems of a few vertices each. Profiling the faithful
implementation at n = 10⁶ counts about 4.3 million queue pulls and 4.3
million base-case calls, and 69 million edge scans where plain Dijkstra
needs 4 million ([BENCHMARKS.md](../BENCHMARKS.md) has the full profile).

**Cache.** Part 1 noted Dijkstra's memory behavior is simple and
prefetch-friendly. BMSSP's recursion hops between many small structures —
exactly the access pattern modern memory hierarchies punish.

None of this is sloppiness. It's what the machinery that *earned* the better
exponent costs at sizes where `log n` is 20 rather than 20,000.

## The measurements

Same machine, same graphs, distances cross-checked bit-for-bit between
implementations (all numbers below are in
[`benchmarks/results/`](../benchmarks/results/), with the methodology in
[BENCHMARKS.md](../BENCHMARKS.md)):

| graph | Dijkstra (this repo) | faithful BMSSP |
|---|---:|---:|
| random, n = 10⁶, m = 4n | 0.85 s | 24.6 s — **29× slower** |
| USA-road-d.NY (264k vertices) | 26 ms | 1.65 s — **64× slower** |

The gap does narrow as `n` grows — 84×, 45×, 29×, 26× at n = 10⁴ through
10⁷ — in the direction, and roughly at the rate, the `log^(1/3) n` scaling
predicts; force the theory's `C·log₂(n)^(−1/3)` form onto the points and it
fits, but only with `C` drifting rather than held constant
([BENCHMARKS.md](../BENCHMARKS.md) shows the fit). The theory has the trend
right. Extrapolate it to the crossover point, though, and the curves meet
around **n ≈ 2^400,000** — not a large graph but an impossible one: storing
even one vertex per atom would exhaust the observable universe (roughly
2^266 atoms) without making a dent in that exponent. The asymptotic
advantage is real and it is unreachable.

A fair objection: maybe the *faithful* implementation is just naive, and an
engineered one would win? This repository spent two optimization passes
finding out ([VARIANTS.md](../VARIANTS.md),
[OPTIMIZATION.md](../OPTIMIZATION.md)): deleting every cost the correctness
proofs permit deleting — the transform, the paper's parameter choices, hash
tables, bounds checks, memory layout, all of it measured change by change.
The end state, `bmssp-fast`, runs about 1.1–1.2× of Dijkstra's time on random
graphs at n = 10⁶–10⁷ and ~2× on the NY road network, where integer weights
make path-length ties everywhere and bmssp-fast pays for each one (more on
ties below) — close enough to touch, and ahead of every published
implementation we could find (an independent C++ study,
[arXiv:2511.03007](https://arxiv.org/abs/2511.03007), reports ~3.6× for its
best variant; a Rust study on Lightning Network graphs,
[arXiv:2509.13448](https://arxiv.org/abs/2509.13448), lands at ~2× at
best). But the profile shows *why* it got that
close: at its measured-optimal settings, the framework collapses into a
single Dijkstra-like pass carrying BMSSP's heavier labels. Every knob, turned
toward "faster," turned toward "more like Dijkstra." The residual gap is the
per-edge price of BMSSP's bookkeeping, and per-edge prices don't fade as `n`
grows. No crossover — just a smaller loss.

One more lesson the stopwatch taught that theory never would have: an
innocent-looking rule in the paper — relax on *less-than-or-equal*, not just
less-than — turns out to interact with equal-length paths so that a naive
implementation can melt down on real road networks, where ties are
everywhere. Random test graphs never trip it, because random real-valued
weights never tie exactly. The write-up is in
[OPTIMIZATION.md](../OPTIMIZATION.md); finding it required running on real
data.

## What to take away

- **Asymptotic analysis and benchmarking answer different questions.** "Whose
  curve wins eventually?" and "what should my code call?" have different
  answers here, and both are correct.
- **Constants are physics, not noise.** Cache lines, allocation churn, and
  per-edge instruction counts decide real races at real sizes.
- **A negative result, measured honestly, is a result.** Knowing that the
  sorting barrier can be broken in theory *and* that breaking it buys nothing
  at feasible sizes — with the numbers and verification to back both halves —
  is worth more than either claim alone.

And none of this diminishes the theorem. Breaking the sorting barrier was a
question open for decades; Duan, Mao, Mao, Shu and Yin
([arXiv:2504.17033](https://arxiv.org/abs/2504.17033)) settled it. Theorems
are forever; constants are negotiable — someone may yet engineer them down.
The two kinds of progress just shouldn't be confused for one another, and a
repository like this one exists to keep the second kind honest.

*End of the tour. Where to next: the same story at full technical depth runs
[ALGORITHM.md](../ALGORITHM.md) → [VARIANTS.md](../VARIANTS.md) →
[OPTIMIZATION.md](../OPTIMIZATION.md) → [BENCHMARKS.md](../BENCHMARKS.md);
the code starts at [`src/dijkstra.rs`](../src/dijkstra.rs) and
[`src/bmssp.rs`](../src/bmssp.rs); and `pip`-facing API docs are in the
[README](../README.md).*
