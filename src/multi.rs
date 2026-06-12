//! Parallel multi-source shortest paths: one independent single-source run
//! per source, fanned out over rayon's thread pool.
//!
//! Each worker writes its own disjoint row of the flat `k x n` output
//! matrices, so the parallelism cannot change any result: row `i` is
//! bit-identical to what the corresponding single-source call produces
//! (BMSSP rows also use the same per-source seed either way).

use crate::bmssp::{sssp_bmssp, BmsspError, Csr};
use crate::dijkstra::{self, DijkstraError};
use rayon::prelude::*;

/// Errors of the multi-source entry points.
pub enum MultiError {
    /// A source index is out of range (its value is reported).
    SourceOutOfRange(i64),
    /// Negative/NaN edge weight.
    BadWeight,
    /// Malformed CSR structure.
    MalformedCsr,
}

/// Run Dijkstra from every source in `sources` in parallel, filling row `i`
/// of `dist`/`pred` (both of length `sources.len() * n`) with the result for
/// `sources[i]`. Rows are bit-identical to single-source [`dijkstra::dijkstra`]
/// calls.
pub fn dijkstra_multi(
    indptr: &[i64],
    indices: &[i32],
    weights: &[f64],
    sources: &[i64],
    n: usize,
    dist: &mut [f64],
    pred: &mut [i32],
) -> Result<(), MultiError> {
    debug_assert_eq!(dist.len(), sources.len() * n);
    debug_assert_eq!(pred.len(), sources.len() * n);
    for &s in sources {
        if s < 0 || (s as usize) >= n {
            return Err(MultiError::SourceOutOfRange(s));
        }
    }

    dist.par_chunks_exact_mut(n)
        .zip_eq(pred.par_chunks_exact_mut(n))
        .zip_eq(sources.par_iter())
        .try_for_each(|((drow, prow), &src)| {
            // Per-worker heap; reused across nothing (one run per row), but
            // its capacity is reserved once inside `dijkstra`.
            let mut heap = dijkstra::Heap::new();
            dijkstra::dijkstra(indptr, indices, weights, src as usize, drow, prow, &mut heap)
                .map_err(|e| match e {
                    DijkstraError::NegativeWeight => MultiError::BadWeight,
                    DijkstraError::MalformedCsr => MultiError::MalformedCsr,
                })
        })
}

/// Run BMSSP from every source in `sources` in parallel (same contract as
/// [`dijkstra_multi`]; row `i` is bit-identical to a single-source
/// [`sssp_bmssp`] call with the same `seed`).
///
/// Note: each worker holds its own transformed graph and run state, so peak
/// memory is roughly `min(len(sources), n_threads)` times that of one
/// single-source BMSSP run.
pub fn bmssp_multi(
    g: &Csr,
    sources: &[i64],
    seed: u64,
    dist: &mut [f64],
    pred: &mut [i32],
) -> Result<(), MultiError> {
    let n = g.n;
    debug_assert_eq!(dist.len(), sources.len() * n);
    for &s in sources {
        if s < 0 || (s as usize) >= n {
            return Err(MultiError::SourceOutOfRange(s));
        }
    }

    dist.par_chunks_exact_mut(n)
        .zip_eq(pred.par_chunks_exact_mut(n))
        .zip_eq(sources.par_iter())
        .try_for_each(|((drow, prow), &src)| {
            let run = sssp_bmssp(g, src as usize, seed, None).map_err(|e| match e {
                BmsspError::SourceOutOfRange => MultiError::SourceOutOfRange(src),
                BmsspError::BadWeight => MultiError::BadWeight,
            })?;
            drow.copy_from_slice(&run.dist);
            prow.copy_from_slice(&run.pred);
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_queue::SplitMix64;
    use crate::bmssp::build_csr;

    fn random_graph(n: usize, m: usize, seed: u64) -> Vec<(u32, u32, f64)> {
        let mut rng = SplitMix64::new(seed);
        (0..m)
            .map(|_| {
                let u = (rng.next_u64() % n as u64) as u32;
                let v = (rng.next_u64() % n as u64) as u32;
                let w = ((rng.next_u64() % 1000) + 1) as f64 / 1000.0;
                (u, v, w)
            })
            .collect()
    }

    /// CSR triple in the borrowed (i64/i32/f64) layout `dijkstra` takes.
    fn as_arrays(g: &Csr) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        (
            g.indptr.iter().map(|&p| p as i64).collect(),
            g.indices.iter().map(|&v| v as i32).collect(),
            g.weights.clone(),
        )
    }

    #[test]
    fn rows_match_single_source_dijkstra() {
        let n = 300;
        let g = build_csr(n, &random_graph(n, 1200, 99));
        let (indptr, indices, weights) = as_arrays(&g);
        let sources: Vec<i64> = vec![0, 7, 299, 7];

        let mut dist = vec![0.0; sources.len() * n];
        let mut pred = vec![0i32; sources.len() * n];
        assert!(dijkstra_multi(&indptr, &indices, &weights, &sources, n, &mut dist, &mut pred)
            .is_ok());

        for (i, &s) in sources.iter().enumerate() {
            let mut d1 = vec![0.0; n];
            let mut p1 = vec![0i32; n];
            let mut heap = dijkstra::Heap::new();
            dijkstra::dijkstra(&indptr, &indices, &weights, s as usize, &mut d1, &mut p1, &mut heap)
                .ok()
                .unwrap();
            assert_eq!(&dist[i * n..(i + 1) * n], &d1[..], "dist row {i}");
            assert_eq!(&pred[i * n..(i + 1) * n], &p1[..], "pred row {i}");
        }
    }

    #[test]
    fn rows_match_single_source_bmssp() {
        let n = 200;
        let g = build_csr(n, &random_graph(n, 800, 7));
        let sources: Vec<i64> = vec![3, 0, 150];

        let mut dist = vec![0.0; sources.len() * n];
        let mut pred = vec![0i32; sources.len() * n];
        assert!(bmssp_multi(&g, &sources, 0x5EED, &mut dist, &mut pred).is_ok());

        for (i, &s) in sources.iter().enumerate() {
            let run = sssp_bmssp(&g, s as usize, 0x5EED, None).unwrap();
            assert_eq!(&dist[i * n..(i + 1) * n], &run.dist[..], "dist row {i}");
            assert_eq!(&pred[i * n..(i + 1) * n], &run.pred[..], "pred row {i}");
        }
    }

    #[test]
    fn bad_source_rejected() {
        let n = 10;
        let g = build_csr(n, &random_graph(n, 30, 1));
        let (indptr, indices, weights) = as_arrays(&g);
        let mut dist = vec![0.0; n];
        let mut pred = vec![0i32; n];
        assert!(matches!(
            dijkstra_multi(&indptr, &indices, &weights, &[10], n, &mut dist, &mut pred),
            Err(MultiError::SourceOutOfRange(10))
        ));
        assert!(matches!(
            bmssp_multi(&g, &[-1], 0, &mut dist, &mut pred),
            Err(MultiError::SourceOutOfRange(-1))
        ));
    }
}
