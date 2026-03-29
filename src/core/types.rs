//! Core data types for genomic island detection.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use serde::{Deserialize, Serialize};

// =============================================================================
// GenomicIsland
// =============================================================================

/// Represents a detected genomic island with its properties.
///
/// A genomic island is a region of atypical composition that may indicate
/// horizontal gene transfer (HGT) or other evolutionary events.
#[cfg_attr(feature = "python", pyclass(get_all, frozen))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenomicIsland {
    /// Start position in the sequence (0-indexed)
    pub start: usize,
    /// End position in the sequence (exclusive)
    pub end: usize,
    /// Overall deviation score (higher = more atypical)
    pub score: f64,
    /// GC content of the island region
    pub gc_content: f64,
    /// K-mer compositional deviation
    pub kmer_deviation: f64,
    /// Identifier of the source sequence
    pub sequence_id: String,
}

impl GenomicIsland {
    /// Create a new genomic island.
    pub fn new(
        start: usize,
        end: usize,
        score: f64,
        gc_content: f64,
        kmer_deviation: f64,
        sequence_id: String,
    ) -> Self {
        GenomicIsland {
            start,
            end,
            score,
            gc_content,
            kmer_deviation,
            sequence_id,
        }
    }

    /// Get the length of the island in base pairs.
    pub fn length(&self) -> usize {
        self.end - self.start
    }

    /// Get a string representation.
    pub fn display(&self) -> String {
        format!(
            "GenomicIsland(start={}, end={}, score={:.4}, gc={:.2}%, seq='{}')",
            self.start,
            self.end,
            self.score,
            self.gc_content * 100.0,
            self.sequence_id
        )
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GenomicIsland {
    #[new]
    fn py_new(
        start: usize,
        end: usize,
        score: f64,
        gc_content: f64,
        kmer_deviation: f64,
        sequence_id: String,
    ) -> Self {
        Self::new(start, end, score, gc_content, kmer_deviation, sequence_id)
    }

    fn __repr__(&self) -> String {
        self.display()
    }

    #[pyo3(name = "length")]
    fn py_length(&self) -> usize {
        self.length()
    }
}

// =============================================================================
// WindowScore
// =============================================================================

/// Window analysis result for a single sliding window.
///
/// Contains the deviation metrics computed for a window at a specific position.
#[cfg_attr(feature = "python", pyclass(get_all, frozen))]
#[derive(Clone, Debug)]
pub struct WindowScore {
    /// Position of the window start in the sequence (0-indexed)
    pub position: usize,
    /// Combined deviation score
    pub score: f64,
    /// GC content of this window
    pub gc_content: f64,
    /// K-mer compositional deviation for this window
    pub kmer_deviation: f64,
}

impl WindowScore {
    /// Create a new window score.
    pub fn new(position: usize, score: f64, gc_content: f64, kmer_deviation: f64) -> Self {
        WindowScore {
            position,
            score,
            gc_content,
            kmer_deviation,
        }
    }

    /// Get a string representation.
    pub fn display(&self) -> String {
        format!(
            "WindowScore(pos={}, score={:.4}, gc={:.2}%)",
            self.position,
            self.score,
            self.gc_content * 100.0
        )
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl WindowScore {
    #[new]
    fn py_new(position: usize, score: f64, gc_content: f64, kmer_deviation: f64) -> Self {
        Self::new(position, score, gc_content, kmer_deviation)
    }

    fn __repr__(&self) -> String {
        self.display()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genomic_island_struct() {
        let island = GenomicIsland::new(1000, 5000, 0.15, 0.55, 0.12, "seq1".to_string());
        assert_eq!(island.start, 1000);
        assert_eq!(island.end, 5000);
        assert_eq!(island.length(), 4000);
    }

    #[test]
    fn test_window_score_struct() {
        let score = WindowScore::new(500, 0.08, 0.52, 0.06);
        assert_eq!(score.position, 500);
        assert!((score.score - 0.08).abs() < 0.001);
    }

    #[test]
    fn test_genomic_island_display() {
        let island = GenomicIsland::new(0, 1000, 0.1, 0.5, 0.05, "test".to_string());
        let display = island.display();
        assert!(display.contains("GenomicIsland"));
        assert!(display.contains("start=0"));
    }
}
