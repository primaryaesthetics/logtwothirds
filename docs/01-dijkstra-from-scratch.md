# Dijkstra from scratch

*Part 1 of the guided tour. Builds on [part 0](00-why-shortest-paths.md):
graphs, weights, and the "shortest paths are made of shortest paths" idea.*

## The wave picture

Imagine pouring water on the source vertex and letting it spread along the
edges at one unit of distance per second. The water reaches each vertex
exactly at that vertex's shortest distance from the source. Dijkstra's
algorithm is a discrete simulation of this wave.

At any moment, every vertex is in one of three states:

- **settled** — the wave has reached it; its distance is final and known;
- **frontier** — a neighbor of the settled region; we've seen *a* way to
  reach it, so it has a tentative distance, which might still improve;
- **unseen** — the wave hasn't come near it yet.

The algorithm is one loop: repeatedly take the frontier vertex with the
*smallest* tentative distance, declare it settled, and update its neighbors.

## Why the smallest one is safe to settle

This is the greedy step, and it deserves a picture before a proof. Think of
the settled region as an island. Every route we're pricing starts at the
source, and the source is on the island — so any route to a not-yet-settled
vertex must step off the island at some point, and the first vertex it
touches when it does is, by definition, a frontier vertex.

Now let `u` be the frontier vertex with the smallest tentative distance `d`,
and take any route to `u` the algorithm hasn't discovered yet. Like every
route, it steps off the island at some frontier vertex `v`. Reaching `v`
already costs at least `d` — we picked `u` precisely because no frontier
vertex is cheaper to reach. And the rest of the trip, from `v` onward to
`u`, can only add to that total: weights are non-negative. So every route to
`u` costs at least `d`, the tentative distance is final, and `u` is safe to
settle.

Notice where the assumption earned its keep: with a negative edge, the "can
only add to that total" step breaks, and so does the whole algorithm. That's
why part 0 insisted on non-negative weights.

## A worked example

Source is A. Edge weights on the arrows:

```
        4         3
   A ------- B ------- E
   |         |
 1 |         | 5
   |    2    |
   C ------- D
```

| step | settled so far | frontier (tentative)     | action                       |
|------|----------------|--------------------------|------------------------------|
| 0    | —              | A(0)                     | settle A, relax its edges    |
| 1    | A              | C(1), B(4)               | settle C — smallest          |
| 2    | A, C           | D(1+2=3), B(4)           | settle D                     |
| 3    | A, C, D        | B(4)*                    | settle B                     |
| 4    | A, C, D, B     | E(4+3=7)                 | settle E                     |
| 5    | all            | —                        | done                         |

*At step 3, D offers B a route costing 3+5=8; B already has 4, so the offer
is rejected. That update-if-better step is called **relaxation**: for each
edge out of a freshly settled `u`, check whether `dist(u) + weight` improves
the neighbor's tentative distance, and record the improvement if so.

Final distances from A: A=0, C=1, D=3, B=4, E=7. Read the settle order back:
A, C, D, B, E — the vertices came out **sorted by distance**. Hold that
thought; it is the entire subject of part 2.

## The cost, and where it goes

Two kinds of work happen:

1. **Every edge is relaxed once** — when its tail vertex is settled. That's
   unavoidable-looking: an algorithm that never looks at some edge can't know
   the edge wasn't a shortcut. Cost: proportional to `m`, the number of
   edges.
2. **Finding the frontier minimum**, once per vertex. Scanning the whole
   frontier each time is slow, so real implementations keep the frontier in a
   **priority queue** (a *heap*) — a structure that answers "what's the
   current minimum?" cheaply. Each heap operation costs about `log n` steps
   for a frontier of size up to `n`, and we do it `n` times.

Total: roughly `m + n log n` operations. In the notation you'll see
everywhere, `O(m + n log n)`. For a graph with a million vertices,
`log₂ n ≈ 20` — the heap adds a factor-twenty-ish tax on the vertex work,
and that's it.

Two things make Dijkstra brutally fast on real machines, beyond the formula.
The per-step work is tiny — a compare, an add, a couple of memory reads. And
the memory it touches is predictable, so the CPU's caches and prefetchers do
their job. Keep both points in mind for part 4, where they turn out to decide
everything.

If you want to see a tuned implementation, this repository's production
Dijkstra is in [`src/dijkstra.rs`](../src/dijkstra.rs) — about 300 lines
including documentation, and the documentation explains each trick.

---

*Next: [Part 2 — the sorting barrier](02-the-sorting-barrier.md)*
