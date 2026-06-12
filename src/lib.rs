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

    /// The native extension module `logtwothirds._logtwothirds`.
    #[pymodule]
    fn _logtwothirds(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(dijkstra_py, m)?)?;
        m.add_function(wrap_pyfunction!(bmssp_py, m)?)?;
        m.add_function(wrap_pyfunction!(bmssp_instrumented_py, m)?)?;
        Ok(())
    }
}
