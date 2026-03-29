//! Main predictor class for genome island detection.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;

use thiserror::Error;

use super::ivom::{IVOMAnalysis, IvomError};
use super::kmer::KmerProfile;
use super::scanner::{ScannerError, SlidingWindowScanner};
use super::seq::{clean_sequence, gc_content, SeqError};
use super::types::{GenomicIsland, WindowScore};

// =============================================================================
// Errors
// =============================================================================

#[derive(Error, Debug)]
pub enum PredictorError {
    #[error("Scanner error: {0}")]
    ScannerError(#[from] ScannerError),
    #[error("IVOM error: {0}")]
    IvomError(#[from] IvomError),
    #[error("Sequence error: {0}")]
    SeqError(#[from] SeqError),
}

#[cfg(feature = "python")]
impl From<PredictorError> for PyErr {
    fn from(err: PredictorError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// =============================================================================
// GenomeIslandPredictor
// =============================================================================

/// Main predictor class for genome island detection.
///
/// Combines k-mer profiling, IVOM analysis, and sliding window scanning
/// to identify regions of atypical composition that may indicate
/// horizontal gene transfer events.
/// TD: options for selecting different methods, scoring thresholds, etc.
#[cfg_attr(feature = "python", pyclass(subclass))]
#[derive(Clone)]
pub struct GenomeIslandPredictor {
    k: usize,
    window_size: usize,
    step_size: usize,
    min_island_size: usize,
    score_threshold: f64,
    ivom: IVOMAnalysis,
}

impl GenomeIslandPredictor {
    /// Create a new genome island predictor.
    pub fn new(
        k: usize,
        window_size: usize,
        step_size: usize,
        min_island_size: usize,
        score_threshold: f64,
    ) -> Result<Self, PredictorError> {
        let ivom = IVOMAnalysis::new(8, 1)?;

        Ok(GenomeIslandPredictor {
            k,
            window_size,
            step_size,
            min_island_size,
            score_threshold,
            ivom,
        })
    }

    /// Build background model from the genome sequence.
    pub fn fit(&mut self, sequence: &str) -> Result<(), PredictorError> {
        self.ivom
            .build_background_model(vec![sequence.to_string()])?;
        Ok(())
    }

    /// Get score threshold.
    pub fn score_threshold(&self) -> f64 {
        self.score_threshold
    }

    /// Set score threshold.
    pub fn set_score_threshold(&mut self, value: f64) {
        self.score_threshold = value;
    }

    /// Get minimum island size.
    pub fn min_island_size(&self) -> usize {
        self.min_island_size
    }

    /// Set minimum island size.
    pub fn set_min_island_size(&mut self, value: usize) {
        self.min_island_size = value;
    }

    /// Predict genomic islands in a sequence (Rust API).
    pub fn predict(
        &self,
        sequence: &str,
        sequence_id: Option<&str>,
    ) -> Result<Vec<GenomicIsland>, PredictorError> {
        let seq_id = sequence_id.unwrap_or("unknown").to_string();
        let cleaned = clean_sequence(sequence.as_bytes())?;
        let seq_bytes = cleaned.as_bytes();

        let background_profile = KmerProfile::from_sequence(seq_bytes, self.k);
        let genome_gc = gc_content(seq_bytes);

        let scanner = SlidingWindowScanner::new(self.window_size, self.step_size, self.k)?;
        let deviations = scanner.scan_with_background(&cleaned, &background_profile)?;

        let islands = self.identify_islands(&deviations, seq_bytes, genome_gc, &seq_id);

        Ok(islands)
    }

    /// Predict islands and return detailed window scores (Rust API).
    pub fn predict_with_scores(
        &self,
        sequence: &str,
        sequence_id: Option<&str>,
    ) -> Result<(Vec<GenomicIsland>, Vec<WindowScore>), PredictorError> {
        let seq_id = sequence_id.unwrap_or("unknown").to_string();
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

        let deviations: Vec<(usize, f64)> = scores.iter().map(|s| (s.position, s.score)).collect();
        let islands = self.identify_islands(&deviations, seq_bytes, genome_gc, &seq_id);

        Ok((islands, scores))
    }

    /// Identify contiguous island regions from window scores.
    fn identify_islands(
        &self,
        deviations: &[(usize, f64)],
        seq_bytes: &[u8],
        genome_gc: f64,
        seq_id: &str,
    ) -> Vec<GenomicIsland> {
        let mut islands = Vec::new();
        let mut in_island = false;
        let mut island_start = 0;
        let mut island_scores = Vec::new();

        for &(pos, score) in deviations {
            if score > self.score_threshold {
                if !in_island {
                    in_island = true;
                    island_start = pos;
                    island_scores.clear();
                }
                island_scores.push(score);
            } else if in_island {
                let island_end = pos;
                let island_size = island_end - island_start + self.window_size;

                if island_size >= self.min_island_size {
                    let island_seq = &seq_bytes[island_start..island_end.min(seq_bytes.len())];
                    let island_gc = gc_content(island_seq);
                    let avg_score: f64 =
                        island_scores.iter().sum::<f64>() / island_scores.len() as f64;
                    let kmer_dev = avg_score - 0.3 * (island_gc - genome_gc).abs();

                    islands.push(GenomicIsland::new(
                        island_start,
                        island_end.min(seq_bytes.len()),
                        avg_score,
                        island_gc,
                        kmer_dev,
                        seq_id.to_string(),
                    ));
                }
                in_island = false;
            }
        }

        // Handle island extending to end
        if in_island && !island_scores.is_empty() {
            let island_end = seq_bytes.len();
            let island_size = island_end - island_start;

            if island_size >= self.min_island_size {
                let island_seq = &seq_bytes[island_start..island_end];
                let island_gc = gc_content(island_seq);
                let avg_score: f64 =
                    island_scores.iter().sum::<f64>() / island_scores.len() as f64;
                let kmer_dev = avg_score - 0.3 * (island_gc - genome_gc).abs();

                islands.push(GenomicIsland::new(
                    island_start,
                    island_end,
                    avg_score,
                    island_gc,
                    kmer_dev,
                    seq_id.to_string(),
                ));
            }
        }

        self.merge_nearby_islands(islands)
    }

    fn merge_nearby_islands(&self, mut islands: Vec<GenomicIsland>) -> Vec<GenomicIsland> {
        if islands.len() <= 1 {
            return islands;
        }

        islands.sort_by_key(|i| i.start);
        let merge_distance = self.window_size * 2;

        let mut merged = Vec::new();
        let mut current = islands.remove(0);

        for next in islands {
            if next.start <= current.end + merge_distance {
                current = GenomicIsland::new(
                    current.start,
                    next.end,
                    (current.score + next.score) / 2.0,
                    (current.gc_content + next.gc_content) / 2.0,
                    (current.kmer_deviation + next.kmer_deviation) / 2.0,
                    current.sequence_id.clone(),
                );
            } else {
                merged.push(current);
                current = next;
            }
        }
        merged.push(current);

        merged
    }
}

// Python-specific methods for access (can probably combine with Rust methods to avoid dups)
#[cfg(feature = "python")]
#[pymethods]
impl GenomeIslandPredictor {
    #[new]
    #[pyo3(signature = (k=4, window_size=5000, step_size=1000, min_island_size=8000, score_threshold=0.1))]
    fn py_new(
        k: usize,
        window_size: usize,
        step_size: usize,
        min_island_size: usize,
        score_threshold: f64,
    ) -> PyResult<Self> {
        Self::new(k, window_size, step_size, min_island_size, score_threshold).map_err(|e| e.into())
    }

    #[pyo3(name = "fit")]
    fn py_fit(&mut self, sequence: &str) -> PyResult<()> {
        self.fit(sequence).map_err(|e| e.into())
    }

    #[getter]
    fn get_score_threshold(&self) -> f64 {
        self.score_threshold
    }

    #[setter]
    fn set_score_threshold_py(&mut self, value: f64) {
        self.score_threshold = value;
    }

    #[getter]
    fn get_min_island_size(&self) -> usize {
        self.min_island_size
    }

    #[setter]
    fn set_min_island_size_py(&mut self, value: usize) {
        self.min_island_size = value;
    }

    /// Predict genomic islands in a sequence (Python API).
    #[pyo3(name = "predict")]
    #[pyo3(signature = (sequence, sequence_id=None))]
    fn py_predict(
        &self,
        py: Python,
        sequence: &str,
        sequence_id: Option<&str>,
    ) -> PyResult<Py<PyList>> {
        let islands = self.predict(sequence, sequence_id)?;

        let list = PyList::new(
            py,
            islands
                .into_iter()
                .map(|i| i.into_pyobject(py).unwrap()),
        )?;
        Ok(list.into())
    }

    /// Predict islands and return detailed window scores (Python).
    #[pyo3(name = "predict_with_scores")]
    #[pyo3(signature = (sequence, sequence_id=None))]
    fn py_predict_with_scores(
        &self,
        py: Python,
        sequence: &str,
        sequence_id: Option<&str>,
    ) -> PyResult<(Py<PyList>, Py<PyList>)> {
        let (islands, scores) = self.predict_with_scores(sequence, sequence_id)?;

        let islands_list = PyList::new(
            py,
            islands
                .into_iter()
                .map(|i| i.into_pyobject(py).unwrap()),
        )?;
        let scores_list = PyList::new(
            py,
            scores
                .into_iter()
                .map(|s| s.into_pyobject(py).unwrap()),
        )?;

        Ok((islands_list.into(), scores_list.into()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_creation() {
        let predictor = GenomeIslandPredictor::new(4, 5000, 1000, 8000, 0.1).unwrap();
        assert_eq!(predictor.score_threshold(), 0.1);
        assert_eq!(predictor.min_island_size(), 8000);
    }

    #[test]
    fn test_predictor_fit() {
        let mut predictor = GenomeIslandPredictor::new(4, 100, 50, 200, 0.1).unwrap();
        let seq = "ATGC".repeat(100);
        assert!(predictor.fit(&seq).is_ok());
    }
}
