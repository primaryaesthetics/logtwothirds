//! logtwothirds core: Dijkstra and BMSSP single-source shortest paths.
//!
//! The pure-Rust algorithms live in [`dijkstra`], [`bmssp`], and
//! [`block_queue`]; the PyO3 bindings (module `logtwothirds._logtwothirds`)
//! are gated behind the `python` feature so that `cargo test` / `cargo
//! clippy` never need a Python interpreter to link. maturin builds with
//! `--features python` (see `pyproject.toml`).
//!
//! The CSR arrays (`indptr: int64`, `indices: int32`, `weights: float64`) are
//! borrowed from NumPy **zero-copy** via rust-numpy `PyReadonlyArray1`.
//! Outputs are freshly allocated NumPy arrays returned to Python.

pub mod block_queue;
pub mod bmssp;
pub mod dijkstra;
pub mod multi;
pub mod variants;

/// BMSSP makes tens of millions of small short-lived allocations per run;
/// mimalloc serves them far faster than the default system heap (especially
/// on Windows). Purely an allocator swap — observable results are unchanged.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "python")]
mod python {
    use crate::bmssp::{sssp_bmssp, BmsspError, Csr};
    use crate::dijkstra::{self, DijkstraError};
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::{PyIndexError, PyValueError};
    use pyo3::prelude::*;

    /// Return type of [`dijkstra_py`] / [`bmssp_py`]:
    /// `(distances: float64[n], predecessors: int32[n])`.
    type SsspResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>);

    /// [`bmssp_instrumented_py`] additionally returns the settlement log
    /// `(vertices: int64[s], dhat: float64[s])` over transformed vertex ids.
    type InstrumentedResult<'py> = (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    );

    /// Validate the borrowed CSR triple and source, mirroring the checks in
    /// `dijkstra_py`. Returns `n`.
    fn check_csr(
        indptr: &[i64],
        indices: &[i32],
        weights: &[f64],
        source: i64,
    ) -> PyResult<usize> {
        if indptr.is_empty() {
            return Err(PyValueError::new_err(
                "indptr must have length n + 1 (got empty array)",
            ));
        }
        let n = indptr.len() - 1;
        if source < 0 || (source as usize) >= n {
            return Err(PyIndexError::new_err(format!(
                "source {source} is out of range for graph with {n} vertices",
            )));
        }
        if indices.len() != weights.len() {
            return Err(PyValueError::new_err(
                "indices and weights must have the same length",
            ));
        }
        if (indptr[n] as usize) != indices.len() {
            return Err(PyValueError::new_err("indptr[-1] must equal len(indices)"));
        }
        Ok(n)
    }

    /// Compute single-source shortest paths with Dijkstra's algorithm.
    #[pyfunction]
    #[pyo3(name = "dijkstra")]
    fn dijkstra_py<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        source: i64,
    ) -> PyResult<SsspResult<'py>> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let weights = weights.as_slice()?;
        let n = check_csr(indptr, indices, weights, source)?;

        let mut dist = vec![0.0f64; n];
        let mut pred = vec![0i32; n];
        // The heap reserves its own capacity (nnz + 1) inside `dijkstra`.
        let mut heap = dijkstra::Heap::new();

        let result = py.allow_threads(|| {
            dijkstra::dijkstra(
                indptr,
                indices,
                weights,
                source as usize,
                &mut dist,
                &mut pred,
                &mut heap,
            )
        });

        match result {
            Ok(()) => Ok((dist.into_pyarray(py), pred.into_pyarray(py))),
            Err(DijkstraError::NegativeWeight) => Err(PyValueError::new_err(
                "negative or NaN edge weight encountered; Dijkstra requires weights >= 0",
            )),
            Err(DijkstraError::MalformedCsr) => Err(PyValueError::new_err(
                "malformed CSR: indptr must be non-decreasing, start at 0, end at \
                 len(indices), and every index must be in [0, n)",
            )),
        }
    }

    /// Copy and validate the CSR triple into the owned [`Csr`] the BMSSP core
    /// uses (u32 indices / usize offsets).
    fn to_owned_csr(indptr: &[i64], indices: &[i32], weights: &[f64], n: usize) -> PyResult<Csr> {
        let malformed = || {
            PyValueError::new_err(
                "malformed CSR: indptr must be non-decreasing, start at 0, end at \
                 len(indices), and every index must be in [0, n)",
            )
        };
        let nnz = indices.len();
        if indptr[0] != 0 {
            return Err(malformed());
        }
        let mut prev = 0i64;
        for &p in &indptr[1..] {
            if p < prev || p > nnz as i64 {
                return Err(malformed());
            }
            prev = p;
        }
        let mut idx = Vec::with_capacity(nnz);
        for &v in indices {
            if v < 0 || (v as usize) >= n {
                return Err(malformed());
            }
            idx.push(v as u32);
        }
        Ok(Csr {
            n,
            indptr: indptr.iter().map(|&p| p as usize).collect(),
            indices: idx,
            weights: weights.to_vec(),
        })
    }

    fn map_bmssp_err(e: BmsspError, source: i64, n: usize) -> PyErr {
        match e {
            BmsspError::SourceOutOfRange => PyIndexError::new_err(format!(
                "source {source} is out of range for graph with {n} vertices",
            )),
            BmsspError::BadWeight => PyValueError::new_err(
                "edge weight must be finite and >= 0",
            ),
        }
    }

    /// Compute single-source shortest paths with the BMSSP algorithm
    /// (Duan–Mao–Mao–Shu–Yin) on the constant-degree transform.
    #[pyfunction]
    #[pyo3(name = "bmssp", signature = (indptr, indices, weights, source, seed = 0))]
    fn bmssp_py<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        source: i64,
        seed: u64,
    ) -> PyResult<SsspResult<'py>> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let weights = weights.as_slice()?;
        let n = check_csr(indptr, indices, weights, source)?;
        let g = to_owned_csr(indptr, indices, weights, n)?;

        let result = py.allow_threads(|| sssp_bmssp(&g, source as usize, seed, None));
        let run = result.map_err(|e| map_bmssp_err(e, source, n))?;
        Ok((run.dist.into_pyarray(py), run.pred.into_pyarray(py)))
    }

    /// Like [`bmssp_py`], but also returns the settlement-order log
    /// (transformed-graph vertex ids and their dhat at settlement).
    #[pyfunction]
    #[pyo3(name = "bmssp_instrumented", signature = (indptr, indices, weights, source, seed = 0))]
    fn bmssp_instrumented_py<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        source: i64,
        seed: u64,
    ) -> PyResult<InstrumentedResult<'py>> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let weights = weights.as_slice()?;
        let n = check_csr(indptr, indices, weights, source)?;
        let g = to_owned_csr(indptr, indices, weights, n)?;

        let result = py.allow_threads(|| sssp_bmssp(&g, source as usize, seed, None));
        let run = result.map_err(|e| map_bmssp_err(e, source, n))?;
        let settle_v: Vec<i64> = run.settle_log.iter().map(|&(v, _)| v as i64).collect();
        let settle_d: Vec<f64> = run.settle_log.iter().map(|&(_, d)| d).collect();
        Ok((
            run.dist.into_pyarray(py),
            run.pred.into_pyarray(py),
            settle_v.into_pyarray(py),
            settle_d.into_pyarray(py),
        ))
    }

    /// Run a research variant of BMSSP (`src/variants/`); `variant` is the
    /// name after the "bmssp-" prefix (e.g. "notransform"). `k`/`t` > 0
    /// force the parameters (0 = the variant's default).
    #[pyfunction]
    #[pyo3(name = "bmssp_variant", signature = (indptr, indices, weights, source, variant, seed = 0, k = 0, t = 0))]
    #[allow(clippy::too_many_arguments)]
    fn bmssp_variant_py<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        source: i64,
        variant: &str,
        seed: u64,
        k: usize,
        t: usize,
    ) -> PyResult<SsspResult<'py>> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let weights = weights.as_slice()?;
        let n = check_csr(indptr, indices, weights, source)?;
        let g = to_owned_csr(indptr, indices, weights, n)?;
        let kt = if k > 0 && t > 0 { Some((k, t)) } else { None };

        let result = py.allow_threads(|| {
            crate::variants::run_variant(variant, &g, source as usize, seed, kt)
        });
        match result.map_err(|e| map_bmssp_err(e, source, n))? {
            Some(run) => Ok((run.dist.into_pyarray(py), run.pred.into_pyarray(py))),
            None => Err(PyValueError::new_err(format!(
                "unknown bmssp variant {variant:?}; known: {:?}",
                crate::variants::VARIANT_NAMES
            ))),
        }
    }

    fn map_multi_err(e: crate::multi::MultiError, n: usize) -> PyErr {
        match e {
            crate::multi::MultiError::SourceOutOfRange(s) => PyIndexError::new_err(format!(
                "source {s} is out of range for graph with {n} vertices",
            )),
            crate::multi::MultiError::BadWeight => PyValueError::new_err(
                "negative or NaN edge weight encountered; weights must be >= 0",
            ),
            crate::multi::MultiError::MalformedCsr => PyValueError::new_err(
                "malformed CSR: indptr must be non-decreasing, start at 0, end at \
                 len(indices), and every index must be in [0, n)",
            ),
        }
    }

    /// Multi-source Dijkstra, parallel over sources (rayon). Returns flat
    /// `(distances: float64[k*n], predecessors: int32[k*n])`; row `i` is the
    /// single-source result for `sources[i]` (bit-identical to `dijkstra`).
    /// The Python wrapper reshapes to `(k, n)`.
    #[pyfunction]
    #[pyo3(name = "dijkstra_multisource")]
    fn dijkstra_multisource_py<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        sources: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<SsspResult<'py>> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let weights = weights.as_slice()?;
        let sources = sources.as_slice()?;
        if indptr.is_empty() {
            return Err(PyValueError::new_err(
                "indptr must have length n + 1 (got empty array)",
            ));
        }
        let n = indptr.len() - 1;
        if indices.len() != weights.len() {
            return Err(PyValueError::new_err(
                "indices and weights must have the same length",
            ));
        }
        if (indptr[n] as usize) != indices.len() {
            return Err(PyValueError::new_err("indptr[-1] must equal len(indices)"));
        }

        let mut dist = vec![0.0f64; sources.len() * n];
        let mut pred = vec![0i32; sources.len() * n];
        let result = py.allow_threads(|| {
            crate::multi::dijkstra_multi(indptr, indices, weights, sources, n, &mut dist, &mut pred)
        });
        match result {
            Ok(()) => Ok((dist.into_pyarray(py), pred.into_pyarray(py))),
            Err(e) => Err(map_multi_err(e, n)),
        }
    }

    /// Multi-source BMSSP, parallel over sources (rayon). Same contract as
    /// [`dijkstra_multisource_py`]; every row uses the same pivot `seed` its
    /// single-source `bmssp` call would.
    #[pyfunction]
    #[pyo3(name = "bmssp_multisource", signature = (indptr, indices, weights, sources, seed = 0))]
    fn bmssp_multisource_py<'py>(
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f64>,
        sources: PyReadonlyArray1<'py, i64>,
        seed: u64,
    ) -> PyResult<SsspResult<'py>> {
        let indptr = indptr.as_slice()?;
        let indices = indices.as_slice()?;
        let weights = weights.as_slice()?;
        let sources = sources.as_slice()?;
        if indptr.is_empty() {
            return Err(PyValueError::new_err(
                "indptr must have length n + 1 (got empty array)",
            ));
        }
        let n = indptr.len() - 1;
        if indices.len() != weights.len() {
            return Err(PyValueError::new_err(
                "indices and weights must have the same length",
            ));
        }
        if (indptr[n] as usize) != indices.len() {
            return Err(PyValueError::new_err("indptr[-1] must equal len(indices)"));
        }
        let g = to_owned_csr(indptr, indices, weights, n)?;

        let mut dist = vec![0.0f64; sources.len() * n];
        let mut pred = vec![0i32; sources.len() * n];
        let result = py.allow_threads(|| {
            crate::multi::bmssp_multi(&g, sources, seed, &mut dist, &mut pred)
        });
        match result {
            Ok(()) => Ok((dist.into_pyarray(py), pred.into_pyarray(py))),
            Err(e) => Err(map_multi_err(e, n)),
        }
    }

    /// The native extension module `logtwothirds._logtwothirds`.
    #[pymodule]
    fn _logtwothirds(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(dijkstra_py, m)?)?;
        m.add_function(wrap_pyfunction!(bmssp_py, m)?)?;
        m.add_function(wrap_pyfunction!(bmssp_instrumented_py, m)?)?;
        m.add_function(wrap_pyfunction!(bmssp_variant_py, m)?)?;
        m.add_function(wrap_pyfunction!(dijkstra_multisource_py, m)?)?;
        m.add_function(wrap_pyfunction!(bmssp_multisource_py, m)?)?;
        Ok(())
    }
}
