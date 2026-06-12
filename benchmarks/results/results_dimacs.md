# SSSP benchmark results

- date: 2026-06-12 13:43
- machine: Intel64 Family 6 Model 58 Stepping 9, GenuineIntel / Windows 11
- python: 3.14.3
- numpy/scipy/rustworkx: 2.4.6 / 1.17.1 / 0.17.1
- runs: 5
- warmup: 1
- seeds: random=0xc0ffee ba=0xba0bab bmssp=0
- total wall time: 16 s

## DIMACS USA-road-d.NY

| graph | n | m | lt-dijkstra | lt-bmssp | scipy | rustworkx |
|---|---|---|---|---|---|---|
| USA-road-d.NY | 264,346 | 730,100 | 25.7 ms | 1.74 s | 42.3 ms | 178.4 ms |

Times are the **median of 5 runs** after warmup (`time.perf_counter`, GC off, algorithm call only). “—” = skipped (footnoted above).