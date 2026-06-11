//! PyO3 bindings for logtwothirds.
//!
//! The CSR arrays (`indptr: int64`, `indices: int32`, `weights: float64`) are
//! borrowed from NumPy **zero-copy** via rust-numpy `PyReadonlyArray1`. Outputs
//! are freshly allocated NumPy arrays returned to Python.

mod dijkstra;

use dijkstra::DijkstraError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// Return type of [`dijkstra_py`]: `(distances: float64[n], predecessors: int32[n])`.
type SsspResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>);

/// Compute single-source shortest paths with Dijkstra's algorithm.
///
/// Arguments are raw CSR components plus the source vertex. Returns a tuple of
/// `(distances: float64[n], predecessors: int32[n])`.
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
    let nnz = indices.len();
    if (indptr[n] as usize) != nnz {
        return Err(PyValueError::new_err(
            "indptr[-1] must equal len(indices)",
        ));
    }

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

/// The native extension module `logtwothirds._logtwothirds`.
#[pymodule]
fn _logtwothirds(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dijkstra_py, m)?)?;
    Ok(())
}
