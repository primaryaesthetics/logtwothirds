# Why shortest paths?

*Part 0 of a guided tour. No prior knowledge assumed — if you can read a road
map, you can read this. The tour ends at the question this repository was
built to answer; each part builds only on the parts before it.*

## The problem

You're at home. You want the fastest way to school, and there are several
routes: through the park, along the main road, past the bakery. Each street
segment takes a known amount of time. Which combination of segments is
fastest?

That's the whole problem. It's called **single-source shortest paths** —
"single-source" because everything starts from one place (home), "shortest
paths" because we want the cheapest route from there to every other place, not
just to school. Navigation apps solve it every time you ask for directions.
So do network routers deciding where to send your data, game engines moving
characters around obstacles, and logistics companies planning deliveries.

## Drawing the map as a graph

Strip the map down to what matters: places, and connections between places
with a cost on each. Mathematicians call this a **graph**. The places are
**vertices**, the connections are **edges**, and the cost on each edge is its
**weight** — travel time, distance, money, anything you want to minimize.

```
        4
   A ------- B
   |         |
 1 |         | 5
   |    2    |
   C ------- D
```

Four vertices, four edges. The weight of a path is the sum of its edge
weights: A→B→D costs 4 + 5 = 9, while A→C→D costs 1 + 2 = 3. Same endpoints,
different price. The shortest path from A to D is the C route.

Two details that matter later:

- **Edges can be one-way** (a directed graph). Real streets often are, and
  the algorithm this repository studies is specifically for directed graphs.
- **Weights are never negative.** You can't spend −3 minutes on a street.
  Everything in this tour assumes non-negative weights, and that assumption
  is load-bearing — you'll see exactly where in part 1.

## Why not just try every route?

The obvious plan — list all paths from A to everything, pick the cheapest —
dies immediately. The number of possible paths grows explosively with the
size of the graph: in a dense-enough graph, doubling the number of vertices
can *square* the number of paths. A country-sized road network has more paths
than you could enumerate before the sun burns out. We need something smarter
than brute force.

## The one idea everything else is built on

Here is the observation that makes the problem tractable. Suppose the
shortest path from A to D passes through C. Then the piece of it from A to C
must itself be the shortest path from A to C — because if some other A→C
route were cheaper, you could splice it in and get a cheaper A→D, which
contradicts A→D being shortest.

Shortest paths are made of shortest paths. That means you can build them up
from small to large: first nail down the places closest to the source, then
use those to reach slightly farther places, and so on outward, like a wave.
No path ever needs to be considered twice.

There's a name for strategies like this — **greedy**: at every step, commit
to the option that looks best right now, and never revisit the decision. For
most problems greedy is a heuristic that can go wrong. For shortest paths
with non-negative weights it is provably exact, and the proof is short enough
to fit in the next part.

The algorithm that turns this idea into concrete steps was written down by
Edsger Dijkstra in 1959, fits on an index card, and is still — this is the
punchline of the whole tour, so remember it — extremely hard to beat in
practice. Part 1 builds it from scratch.

---

*Next: [Part 1 — Dijkstra from scratch](01-dijkstra-from-scratch.md)*
