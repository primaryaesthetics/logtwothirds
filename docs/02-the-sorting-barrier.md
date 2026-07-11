# The sorting barrier

*Part 2 of the guided tour. Builds on [part 1](01-dijkstra-from-scratch.md):
Dijkstra settles vertices in order of increasing distance, at cost
`O(m + n log n)`.*

## Dijkstra secretly sorts

Look again at the settle order from part 1's example: A, C, D, B, E: the
vertices in increasing order of distance. That's not a coincidence of the
example; it's the mechanism. Dijkstra always settles the closest unsettled
vertex next, so it *produces a sorted list of all vertices by distance*,
whether you asked for one or not.

Now run that observation in reverse. Take any `n` numbers you'd like to sort.
Build a trivial graph: one source, an edge from the source to vertex `i` with
weight equal to the `i`-th number.

```
        x₁
   s ------→ v₁
   |    x₂
   +--------→ v₂
   |    x₃
   +--------→ v₃         (n spokes)
   ...
```

Run Dijkstra. The order in which it settles `v₁ … vₙ` is exactly the sorted
order of your numbers. So Dijkstra (any algorithm that settles vertices in distance order) contains a sorting algorithm inside it.

## What sorting costs

Sorting has a famous floor. If the only way you may examine values is to
compare two of them ("is x bigger than y?"), then sorting `n` items requires
about `n log n` comparisons in the worst case; no cleverness gets below it.
The argument is a counting one: `n` items can arrive in `n!` possible orders,
each yes/no comparison at best halves the set of orders still possible, and
you need `log₂(n!) ≈ n log n` halvings to pin down one order. This is a
theorem about *every* comparison-based sorting method, including all the ones
nobody has invented yet.

Chain the two facts:

1. Any distance-ordered shortest-path algorithm can be used to sort.
2. Sorting costs at least about `n log n` comparisons.

Therefore any shortest-path algorithm that works like Dijkstra, settling vertices from nearest to farthest, is stuck paying `n log n`. On sparse
graphs, where `m` is comparable to `n`, the `n log n` term dominates the
total cost, so the heap isn't an implementation detail you can optimize away.
It's a wall. People call it the **sorting barrier**, and for decades
`O(m + n log n)` stood as the best known bound for this problem on directed
graphs, so long as you insisted on distance order.

*(Fine print, for honesty: this cost model (comparisons and additions on real-valued weights) is the standard one for the problem, and it's the model
this whole tour lives in. With small integer weights and machine tricks there
are ways around comparisons; the 2025 result is remarkable precisely because
it needs no such assumptions.)*

## The loophole

Look again at fact (1), and at the qualifier hiding inside it:
*distance-ordered*. The problem statement asks for the
*distances*. Nobody asked for them *in sorted order*. The sorting barrier
blocks algorithms that settle nearest-first; it says nothing about an
algorithm that figures out all the distances while never establishing their
full order.

Is that loophole usable? For a long time nobody knew. Computing distances
without ordering them sounds paradoxical: Dijkstra's correctness proof (part
1) leans directly on always knowing the frontier minimum. Give up the sorted
frontier and the greedy argument collapses; you need a different reason to
ever be sure a tentative distance is final.

In 2025, Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu and Longhui Yin published
*"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"*
([arXiv:2504.17033](https://arxiv.org/abs/2504.17033)): a deterministic
algorithm running in `O(m log^(2/3) n)`. For sparse graphs that genuinely
beats `O(m + n log n)`: the exponent on the log dropped below one. The
barrier wasn't broken so much as sidestepped: their algorithm computes every
distance exactly, but only ever learns a *coarse* ordering of the vertices,
never the full sorted one. Just enough order to be correct, provably less
than sorting.

And the story didn't stop there: in February 2026 four of the five authors
sharpened the bound again, to `O(m √(log n · log log n))` for sparse graphs
([arXiv:2602.07868](https://arxiv.org/abs/2602.07868)). This tour follows the
original `log^(2/3)` algorithm (it's the one with published implementations to measure against), but the barrier is now well and truly down.

How can "just enough order" possibly work? That's part 3.

---

*Next: [Part 3 — how BMSSP breaks the barrier](03-how-bmssp-breaks-it.md)*
