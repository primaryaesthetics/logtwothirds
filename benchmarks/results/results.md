# SSSP benchmark results

- date: 2026-06-12 10:55
- machine: Intel64 Family 6 Model 58 Stepping 9, GenuineIntel / Windows 11
- python: 3.14.3
- numpy/scipy/rustworkx: 2.4.6 / 1.17.1 / 0.17.1
- runs: 5
- warmup: 1
- seeds: random=0xc0ffee ba=0xba0bab bmssp=0
- total wall time: 3726 s
- dimacs run date: 2026-06-12 13:43

## Random directed graphs (m = 4n)

| graph | n | m | lt-dijkstra | lt-bmssp | scipy | rustworkx |
|---|---|---|---|---|---|---|
| n=10^4 | 10,000 | 39,991 | 1.2 ms | 137.4 ms | 2.0 ms | 4.9 ms |
| n=10^5 | 100,000 | 399,996 | 40.0 ms | 1.83 s | 42.0 ms | 80.8 ms |
| n=10^6 | 1,000,000 | 3,999,995 | 885.7 ms | 25.55 s | 826.6 ms | 1.48 s |
| n=10^7 | 10,000,000 | 39,999,994 | 14.75 s | 405.14 s | 12.60 s | — |
| | | | *skipped: edge count over --rustworkx-max-edges (PyDiGraph build too large for this machine)* | | | |

## Barabási–Albert graphs (attachment 4, symmetrized)

| graph | n | m | lt-dijkstra | lt-bmssp | scipy | rustworkx |
|---|---|---|---|---|---|---|
| n=10^4 | 10,000 | 79,974 | 2.0 ms | 359.8 ms | 3.5 ms | 10.3 ms |
| n=10^5 | 100,000 | 799,974 | 61.2 ms | 4.22 s | 53.7 ms | 153.0 ms |
| n=10^6 | 1,000,000 | 7,999,974 | 1.35 s | 53.31 s | 1.35 s | 2.24 s |

## DIMACS USA-road-d.NY

| graph | n | m | lt-dijkstra | lt-bmssp | scipy | rustworkx |
|---|---|---|---|---|---|---|
| USA-road-d.NY | 264,346 | 730,100 | 25.7 ms | 1.74 s | 42.3 ms | 178.4 ms |

Times are the **median of 5 runs** after warmup (`time.perf_counter`, GC off, algorithm call only). “—” = skipped (footnoted above).