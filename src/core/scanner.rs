//! Sliding window scanner for analyzing compositional variation along a genome.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;

use thiserror::Error;

use super::kmer::KmerProfile;
use super::seq::{clean_sequence, gc_content, SeqError};
use super::types::WindowScore;

// =============================================================================
// Errors
// =============================================================================

#[derive(Error, Debug)]
pub enum ScannerError {
    #[error("window_size must be >= k-mer size")]
    WindowTooSmall,
    #[error("step_size must be >= 1")]
    InvalidStepSize,
    #[error("k must be >= 1")]
    InvalidK,
    #[error("Sequence error: {0}")]
    SeqError(#[from] SeqError),
}

#[cfg(feature = "python")]
impl From<ScannerError> for PyErr {
    fn from(err: ScannerError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// =============================================================================
// SlidingWindowScanner
// =============================================================================

/// Sliding window scanner for analyzing compositional variation along a genome.
///
/// Scans a sequence with a sliding window, computing deviation metrics for each
/// window position. Used to identify regions of atypical composition.
#[cfg_attr(feature = "python", pyclass(subclass))]
#[derive(Clone, Debug)]
pub struct SlidingWindowScanner {
    window_size: usize,
    step_size: usize,
    k: usize,
}

impl SlidingWindowScanner {
    /// Create a new sliding window scanner.
    pub fn new(window_size: usize, step_size: usize, k: usize) -> Result<Self, ScannerError> {
        if window_size < k {
            return Err(ScannerError::WindowTooSmall);
        }
        if step_size < 1 {
            return Err(ScannerError::InvalidStepSize);
        }
        if k < 1 {
            return Err(ScannerError::InvalidK);
        }

        Ok(SlidingWindowScanner {
            window_size,
            step_size,
            k,
        })
    }

    /// Get window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get step size.
    pub fn step_size(&self) -> usize {
        self.step_size
    }

    /// Get k-mer size.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Scan a sequence and return raw deviation values as (position, score) tuples.
    pub fn scan_deviations(&self, sequence: &str) -> Result<Vec<(usize, f64)>, ScannerError> {
        let cleaned = clean_sequence(sequence.as_bytes())?;
        let seq_bytes = cleaned.as_bytes();

        let background_profile = KmerProfile::from_sequence(seq_bytes, self.k);

        let mut results = Vec::new();
        let mut pos = 0;

        while pos + self.window_size <= seq_bytes.len() {
            let window = &seq_bytes[pos..pos + self.window_size];
            let window_profile = KmerProfile::from_sequence(window, self.k);
            let deviation = window_profile.js_divergence(&background_profile);
            results.push((pos, deviation));
            pos += self.step_size;
        }

        Ok(results)
    }

    /// Scan and return WindowScore objects (Rust API).
    pub fn scan(&self, sequence: &str) -> Result<Vec<WindowScore>, ScannerError> {
        let cleaned = clean_sequence(sequence.as_bytes())?;
        let seq_bytes = cleaned.as_bytes();

        let background_profile = KmerProfile::from_sequence(seq_bytes, self.k);
        let genome_gc = gc_content(seq_bytes);

        let mut scores = Vec::new();
        let mut pos = 0;

        while pos + self.window_size <= seq_bytes.len() {
            let window = &seq_bytes[pos..pos + self.window_size];
            let window_profile = KmerProfile::from_sequence(window, self.k);

            let window_gc = gc_content(window);
            let deviation = window_profile.js_divergence(&background_profile);
            let gc_deviation = (window_gc - genome_gc).abs();
            let score = deviation + 0.3 * gc_deviation;

            scores.push(WindowScore::new(pos, score, window_gc, deviation));
            pos += self.step_size;
        }

        Ok(scores)
    }

    /// Scan with a provided background profile (unexposed).
    pub fn scan_with_background(
        &self,
        sequence: &str,
        background_profile: &KmerProfile,
    ) -> Result<Vec<(usize, f64)>, ScannerError> {
        let cleaned = clean_sequence(sequence.as_bytes())?;
        let seq_bytes = cleaned.as_bytes();

        let mut results = Vec::new();
        let mut pos = 0;

        while pos + self.window_size <= seq_bytes.len() {
            let window = &seq_bytes[pos..pos + self.window_size];
            let window_profile = KmerProfile::from_sequence(window, self.k);
            let deviation = window_profile.js_divergence(background_profile);
            results.push((pos, deviation));
            pos += self.step_size;
        }

        Ok(results)
    }
}

// Python-specific methods
#[cfg(feature = "python")]
#[pymethods]
impl SlidingWindowScanner {
    #[new]
    #[pyo3(signature = (window_size=5000, step_size=1000, k=4))]
    fn py_new(window_size: usize, step_size: usize, k: usize) -> PyResult<Self> {
        Self::new(window_size, step_size, k).map_err(|e| e.into())
    }

    #[getter]
    fn get_window_size(&self) -> usize {
        self.window_size
    }

    #[getter]
    fn get_step_size(&self) -> usize {
        self.step_size
    }

    #[getter]
    fn get_k(&self) -> usize {
        self.k
    }

    /// Scan a sequence and return window scores as Python list.
    #[pyo3(name = "scan")]
    fn py_scan(&self, py: Python, sequence: &str) -> PyResult<Py<PyList>> {
        let scores = self.scan(sequence)?;

        let list = PyList::new(
            py,
            scores
                .into_iter()
                .map(|s| s.into_pyobject(py).unwrap()),
        )?;
        Ok(list.into())
    }

    /// Scan and return raw deviation values.
    #[pyo3(name = "scan_deviations")]
    fn py_scan_deviations(&self, sequence: &str) -> PyResult<Vec<(usize, f64)>> {
        self.scan_deviations(sequence).map_err(|e| e.into())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scanner_creation() {
        let scanner = SlidingWindowScanner::new(100, 10, 4).unwrap();
        assert_eq!(scanner.window_size(), 100);
        assert_eq!(scanner.step_size(), 10);
        assert_eq!(scanner.k(), 4);
    }

    #[test]
    fn test_scanner_invalid_params() {
        // window_size < k
        assert!(SlidingWindowScanner::new(2, 1, 4).is_err());
        // step_size < 1
        assert!(SlidingWindowScanner::new(100, 0, 4).is_err());
        // k < 1
        assert!(SlidingWindowScanner::new(100, 10, 0).is_err());
    }

    #[test]
    fn test_scan_deviations() {
        let scanner = SlidingWindowScanner::new(10, 5, 2).unwrap();
        let seq = "ATGCATGCATGCATGCATGCATGCATGC";
        let results = scanner.scan_deviations(seq).unwrap();
        assert!(!results.is_empty());
    }
}
