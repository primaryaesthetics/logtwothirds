# SSSP benchmark results

- date: 2026-06-13 04:02
- machine: Intel64 Family 6 Model 58 Stepping 9, GenuineIntel / Windows 11
- python: 3.14.3
- numpy/scipy/rustworkx: 2.4.6 / 1.17.1 / 0.17.1
- runs: 5
- warmup: 1
- seeds: random=0xc0ffee ba=0xba0bab bmssp=0
- total wall time: 3489 s

## Random directed graphs (m = 4n)

| graph | n | m | lt-dijkstra | lt-bmssp | lt-bmssp-fast | scipy | rustworkx |
|---|---|---|---|---|---|---|---|
| n=10^4 | 10,000 | 39,991 | 1.4 ms | 117.3 ms | 2.7 ms | 2.7 ms | 5.7 ms |
| n=10^5 | 100,000 | 399,996 | 36.9 ms | 1.66 s | 67.4 ms | 44.6 ms | 87.1 ms |
| n=10^6 | 1,000,000 | 3,999,995 | 854.4 ms | 24.61 s | 1.34 s | 805.5 ms | 1.59 s |
| n=10^7 | 10,000,000 | 39,999,994 | 13.30 s | 345.11 s | 18.40 s | 10.69 s | — |
| | | | *skipped: edge count over --rustworkx-max-edges (PyDiGraph build too large for this machine)* | | | | |

## Barabási–Albert graphs (attachment 4, symmetrized)

| graph | n | m | lt-dijkstra | lt-bmssp | lt-bmssp-fast | scipy | rustworkx |
|---|---|---|---|---|---|---|---|
| n=10^4 | 10,000 | 79,974 | 1.7 ms | 217.3 ms | 4.1 ms | 2.8 ms | 6.5 ms |
| n=10^5 | 100,000 | 799,974 | 40.0 ms | 3.25 s | 95.5 ms | 49.9 ms | 118.4 ms |
| n=10^6 | 1,000,000 | 7,999,974 | 1.28 s | 43.86 s | 1.79 s | 1.06 s | 1.86 s |

## DIMACS USA-road-d.NY

| graph | n | m | lt-dijkstra | lt-bmssp | lt-bmssp-fast | scipy | rustworkx |
|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 264,346 | 730,100 | 25.9 ms | 1.65 s | 130.5 ms | 40.7 ms | 125.8 ms |

Times are the **median of 5 runs** after warmup (`time.perf_counter`, GC off, algorithm call only). “—” = skipped (footnoted above).