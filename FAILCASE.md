# FAILCASE.md — differential-divergence record (Rust port vs `_reference.py`)

**Status: NO FAILING CASE EXISTS.** As of 2026-06-12 (commit `3ebb3b2` +
extended test sweeps), no graph is known on which the Rust port
(`src/bmssp.rs`, `src/block_queue.rs`) diverges from the Python reference
under the pinned-determinism protocol of `tests/diff_driver.py`. This file
records the investigation that established that, and is the designated home
for a minimal reproducing graph if one is ever found (per the Step E
contingency protocol: localize the first divergence in the settlement-log
prefix, record the minimal graph here, stop after 3 failed root-cause
attempts).

## Investigation (2026-06-12)

Trigger: a report that "FAILCASE.md contains a minimal diverging graph with
both settlement logs attached". Findings:

1. No `FAILCASE.md` existed in the working tree, in any commit on any local
   or remote branch (`git log --all -- "*FAILCASE*"` empty; `origin/main` in
   sync), in the stash, or elsewhere on the machine.
2. The Step E acceptance run had passed 200/200 with distances **and**
   settlement order bit-exact, so the contingency that creates this file
   never fired.
3. The working-tree algorithm sources were byte-identical to the audited
   Step E commit (`e59cc47`).

Extended divergence hunt at that commit, all via `tests/differential.rs` +
`tests/diff_driver.py` (bit-exact comparison of distances and settlement
logs):

| Sweep | Seeds | Shape | Result |
|---|---|---|---|
| Acceptance distribution | 0–999 | n ≤ 5000, mixed sizes, ~5% zero-weight edges | 1000/1000 OK |
| Tie-heavy | 1000–1999 | weights on an 8-value grid incl. 0.0, up to 6n edges | 1000/1000 OK |
| Verification regime | 2000–2999 | tie-heavy shapes, forced (k, t) = (2, 2) (the `_small_params` regime where the pivot branch / L22 W-sweep dominate) | 1000/1000 OK |

Plus: `cargo test` green (unit, differential, property-vs-Dijkstra up to
10^6 edges with bit-exact distances, not_dijkstra), `cargo clippy
--all-targets -- -D warnings` clean (with and without `--features python`),
`pytest -q` 140 passed.

## The expected false positive: comparing without pinning

The reference's *semantics* leave two behaviors unspecified, and its
settlement order genuinely depends on both:

* `random.randint` pivot draws in `_select_smallest` (module-global Mersenne
  Twister by default);
* the iteration order of the builtin sets `U` / `result_U` in `bmssp`,
  leaked through `list(result_U)` into the recursion.

A side-by-side run of the Rust port against the **unpatched** reference will
therefore show settlement logs that differ from the first `Pull`-fed
recursion onward. **That is not a port bug** — it is two valid executions of
the same algorithm. Distances still agree (verified: 30/30 tie-heavy graphs
vs the unpatched reference). The valid comparison pins both choices to the
ones the Rust port implements: `ref.random = SplitMix64(seed)` and
`ref.set = OSet` (insertion-ordered), exactly as `tests/diff_driver.py`
does. Any future failing case reported here must include the pinned
`algo_seed` and have been produced under this protocol — otherwise it is the
known false positive above.

## Template for a real failing case (none known)

```
seed / generator: ...           # or explicit n, edge list, source
algo_seed: 0x...                # SplitMix64 seed used by BOTH sides
params: k=.. t=.. L=.. n2=..    # must match between sides first
first divergence: settle index i
python: (v=.., dhat=.., bits=0x..)  preceding context ...
rust:   (v=.., dhat=.., bits=0x..)  preceding context ...
root cause: ...
fix: ...
```
