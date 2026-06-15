# SSSP benchmark results

- date: 2026-06-15 04:31
- machine: arm / Darwin 23.6.0
- python: 3.14.6
- numpy/scipy/rustworkx: 2.4.6 / 1.17.1 / 0.17.1
- runs: 5
- warmup: 1
- seeds: random=0xc0ffee ba=0xba0bab bmssp=0
- total wall time: 1592 s

## Random directed graphs (m = 4n)

| graph | n | m | lt-dijkstra | lt-bmssp | lt-bmssp-fast | scipy | rustworkx |
|---|---|---|---|---|---|---|---|
| n=10^4 | 10,000 | 39,991 | 0.4 ms | 34.7 ms | 1.3 ms | 1.1 ms | 1.6 ms |
| n=10^5 | 100,000 | 399,996 | 6.6 ms | 470.9 ms | 18.0 ms | 14.8 ms | 27.8 ms |
| n=10^6 | 1,000,000 | 3,999,995 | 180.8 ms | 6.04 s | 493.2 ms | 310.4 ms | 535.1 ms |
| n=10^7 | 10,000,000 | 39,999,994 | 3.77 s | 125.04 s | 6.09 s | 4.81 s | — |
| | | | *skipped: edge count over --rustworkx-max-edges (PyDiGraph build too large for this machine)* | | | | |

## Barabási–Albert graphs (attachment 4, symmetrized)

| graph | n | m | lt-dijkstra | lt-bmssp | lt-bmssp-fast | scipy | rustworkx |
|---|---|---|---|---|---|---|---|
| n=10^4 | 10,000 | 79,974 | 0.8 ms | 77.3 ms | 2.0 ms | 1.7 ms | 2.8 ms |
| n=10^5 | 100,000 | 799,974 | 12.1 ms | 1.05 s | 30.6 ms | 23.8 ms | 48.2 ms |
| n=10^6 | 1,000,000 | 7,999,974 | 293.7 ms | 12.60 s | 675.7 ms | 407.3 ms | 734.6 ms |

## DIMACS USA-road-d.NY

| graph | n | m | lt-dijkstra | lt-bmssp | lt-bmssp-fast | scipy | rustworkx |
|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 264,346 | 730,100 | 9.9 ms | 536.6 ms | 61.0 ms | 22.2 ms | 40.0 ms |

Times are the **median of 5 runs** after warmup (`time.perf_counter`, GC off, algorithm call only). “—” = skipped (footnoted above).