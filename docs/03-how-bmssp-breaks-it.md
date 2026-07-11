# How BMSSP breaks the barrier

*Part 3 of the guided tour. Builds on [part 2](02-the-sorting-barrier.md):
the barrier only binds algorithms that settle vertices in fully sorted
distance order, and the 2025 algorithm slips past it by maintaining less
order than that.*

This is the hardest part of the tour, and it's honest about being a sketch:
the real construction has careful invariants that take the paper ten pages
to state. Everything below is true in outline, and where the outline hides
something, it says so. The rigorous version, distilled lemma by lemma, is in
[ALGORITHM.md](../ALGORITHM.md); the algorithm's name in the paper is
**BMSSP** (*bounded multi-source shortest paths*), after the subproblem its
recursion solves.

## Idea 1: sort bands, not vertices

Dijkstra keeps the frontier in a heap, so at every moment it knows the exact
order of *all* frontier vertices. That's where the `log n` per vertex goes;
and part 2 showed that's paying for a full sort.

BMSSP maintains the frontier in a special queue that only keeps vertices in
**coarse groups by distance** (think "everything roughly 0–10 meters out," "roughly 10–20," and so on), with no ordering inside a group. Its `Pull`
operation hands you one batch: some `M` frontier vertices that are guaranteed
to be *the nearest ones as a set*, in no particular order, plus a number `B'`
meaning "everything I gave you is closer than `B'`; everything I kept is not."
Group-level order, at group-level cost: cheaper than a heap's total order by
exactly the margin the whole result comes down to.

## Idea 2: recurse on a band

Given a batch of near vertices and its boundary `B'`, the algorithm now has a
smaller job: *finish everything closer than `B'`, starting from this batch.*
That job is the same shape as the original problem (compute distances outward from a set of sources, up to a distance bound), so it hands the job
to itself, recursively, with a finer notion of "band" one level down. The
recursion bottoms out in tiny subproblems solved by a bounded mini-Dijkstra
(on `k+1`-ish vertices, small enough that its heap tax is negligible).

Levels do get one thing for free that Dijkstra never has: a *bound*. A
level's call may relax an edge and find the result lands beyond its bound
`B'`: the discovery is simply parked in the parent's queue and becomes some
later band's work. No wasted relaxation, no need to order it now.

So the picture is: the top level slices the distance axis into a few fat
bands; each band's call slices its own range into a few thinner bands; and so
on, a few levels deep. Sorting-wise, the algorithm only ever learns which
*band* a vertex falls in at each level. Multiplied out, that's far less
information than the full sorted order, and information is what comparisons
buy. That's the barrier sidestep, made concrete.

## Idea 3: pivots — keeping the frontier from bloating

One failure mode threatens all of this. A band's call starts from a set `S`
of sources. If `S` is huge, and every source spawns its own little shortest
path tree, the bookkeeping per level blows up and eats the savings.

The fix is the paper's Algorithm 1, **FindPivots**. Before recursing, run `k`
rounds of brute-force relaxation from `S` (Bellman–Ford steps: no ordering
at all, just "relax every edge of the current wave, k times"). Two outcomes:

- Some source's shortest-path tree grew to at least `k` vertices in those
  rounds. Trees that big are few (they don't overlap, so a region of `W` vertices fits at most `W/k` of them), and their roots (**pivots**) are
  the only sources the recursion truly needs to worry about.
- Or nothing grew that big: then those `k` rounds already finished every
  vertex of the band closer than `k` hops, and there's nothing left to
  recurse on.

Either way the frontier the recursion carries shrinks by a factor of about
`k`, at the price of `k` brute-force sweeps. Cheap disorder (Bellman–Ford)
is spent to avoid expensive order (heap operations); the same trade as
ideas 1 and 2, applied a third way.

## Why the exponent is exactly 2/3

The construction has two knobs: `k` (how many brute-force rounds / the pivot
threshold) and `t` (how coarse each level's bands are). Every vertex pays
about `k` for the sweeps it appears in; every edge pays about `t` for the
queue work where it crosses a band boundary; and the recursion is
`(log n)/t` levels deep, multiplying the per-level costs. Balancing (
raising `k` and `t` until the sweep cost, the queue cost, and the depth cost
all match) lands at

```
k = (log n)^(1/3),   t = (log n)^(2/3),   depth = (log n)^(1/3)
```

and a total of **`O(m · log^(2/3) n)`**. Against Dijkstra's `m + n log n`
that's a genuine asymptotic win on sparse graphs: exponent 2/3 on the
logarithm instead of 1. (The counting above is the crude version; the paper's
Section 3 does it properly, including a preliminary transform that caps every
vertex's degree so "per vertex" and "per edge" costs can be exchanged.)

## What the sketch hid

Three things, so you know where the real difficulty lives. *Correctness
without total order*: Dijkstra's proof leaned on always knowing the frontier
minimum; BMSSP's substitute is a subtler invariant (every incomplete vertex in a band still has its true distance reachable through the band's sources)
maintained by every piece above. *Ties*: with equal path lengths, "which
band?" needs a tie-breaking order on paths, and getting it wrong breaks the
recursion silently (this repository found and documents real bugs of exactly
that species). *The queue*: a structure with the promised `Pull` /
batch-insert costs (paper's Lemma 3.3) is itself a nontrivial construction,
block-based, and this repository implements it faithfully in
[`src/block_queue.rs`](../src/block_queue.rs).

The full implementation of everything above, in this repository, is
[`src/bmssp.rs`](../src/bmssp.rs), checked line against the paper, and
verified bit-for-bit against an independent reference implementation on
hundreds of graphs (the receipts are in [AUDIT.md](../AUDIT.md) and
[SPEC.md](../SPEC.md)).

So: an asymptotically faster algorithm, deterministic, exact, implemented
and verified. The natural question, the one this repository exists to answer, is whether it's actually *faster*.

---

*Next: [Part 4 — why theory isn't speed](04-why-theory-isnt-speed.md)*
