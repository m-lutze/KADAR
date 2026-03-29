//! K-mer hashing, MinHash sketching, and k-mer profiles - core implementations.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

use ahash::AHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use thiserror::Error;

use super::seq::{canonical_kmer, clean_sequence, reverse_complement, SeqError};

// =============================================================================
// Errors
// =============================================================================

#[derive(Error, Debug)]
pub enum KmerError {
    #[error("k must be positive")]
    InvalidK,
    #[error("Sequence '{0}' already exists")]
    DuplicateSequence(String),
    #[error("Sequence '{0}' not found")]
    SequenceNotFound(String),
    #[error("Sequence error: {0}")]
    SeqError(#[from] SeqError),
}

#[cfg(feature = "python")]
impl From<KmerError> for PyErr {
    fn from(err: KmerError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// =============================================================================
// K-mer Hashing
// =============================================================================

/// Simple k-mer hash for minhash-style operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KmerHash {
    pub hash: u64,
}

impl KmerHash {
    pub fn new(kmer: &[u8]) -> Self {
        let mut hasher = DefaultHasher::new();
        kmer.hash(&mut hasher);
        KmerHash {
            hash: hasher.finish(),
        }
    }

    /// Create hash with canonical k-mer (lexicographically smaller of forward/reverse).
    pub fn canonical(kmer: &[u8]) -> Self {
        let rc = reverse_complement(kmer);
        let canonical = if kmer <= &rc[..] { kmer } else { &rc[..] };
        Self::new(canonical)
    }
}

// =============================================================================
// MinHash Sketching
// =============================================================================

/// MinHash implementation for k-mer sketching.
#[derive(Clone, Debug)]
pub struct MinHash {
    pub ksize: usize,
    pub num_hashes: usize,
    pub hashes: Vec<u64>,
}

impl MinHash {
    pub fn new(ksize: usize, num_hashes: usize) -> Self {
        MinHash {
            ksize,
            num_hashes,
            hashes: Vec::new(),
        }
    }

    pub fn add_sequence(&mut self, seq: &[u8]) {
        let mut new_hashes = Vec::new();

        for i in 0..=(seq.len().saturating_sub(self.ksize)) {
            let kmer = &seq[i..i + self.ksize];
            let hash = KmerHash::canonical(kmer).hash;
            new_hashes.push(hash);
        }

        new_hashes.sort();
        new_hashes.truncate(self.num_hashes);
        self.hashes = new_hashes;
    }

    pub fn jaccard_similarity(&self, other: &MinHash) -> f64 {
        if self.hashes.is_empty() && other.hashes.is_empty() {
            return 1.0;
        }
        if self.hashes.is_empty() || other.hashes.is_empty() {
            return 0.0;
        }

        let mut intersection = 0;
        let mut i = 0;
        let mut j = 0;

        while i < self.hashes.len() && j < other.hashes.len() {
            match self.hashes[i].cmp(&other.hashes[j]) {
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        let union = self.hashes.len() + other.hashes.len() - intersection;
        intersection as f64 / union as f64
    }
}

// =============================================================================
// K-mer Profile
// =============================================================================

/// K-mer frequency profile for a sequence.
#[derive(Clone, Debug, Default)]
pub struct KmerProfile {
    pub counts: AHashMap<Vec<u8>, u64>,
    pub total: u64,
    pub k: usize,
}

impl KmerProfile {
    pub fn new(k: usize) -> Self {
        KmerProfile {
            counts: AHashMap::new(),
            total: 0,
            k,
        }
    }

    /// Build profile from sequence using canonical k-mers.
    pub fn from_sequence(seq: &[u8], k: usize) -> Self {
        let mut profile = Self::new(k);
        profile.add_sequence(seq);
        profile
    }

    pub fn add_sequence(&mut self, seq: &[u8]) {
        if seq.len() < self.k {
            return;
        }

        for i in 0..=(seq.len() - self.k) {
            let kmer = &seq[i..i + self.k];
            if kmer.iter().all(|&b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                let canonical = canonical_kmer(kmer);
                *self.counts.entry(canonical).or_insert(0) += 1;
                self.total += 1;
            }
        }
    }

    /// Get frequency (normalized count) of a k-mer.
    pub fn frequency(&self, kmer: &[u8]) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let canonical = canonical_kmer(kmer);
        let count = self.counts.get(&canonical).copied().unwrap_or(0);
        count as f64 / self.total as f64
    }

    /// Calculate Jensen-Shannon divergence between two profiles.
    pub fn js_divergence(&self, other: &KmerProfile) -> f64 {
        if self.total == 0 || other.total == 0 {
            return 1.0;
        }

        let mut all_kmers: Vec<Vec<u8>> = self.counts.keys().cloned().collect();
        for kmer in other.counts.keys() {
            if !self.counts.contains_key(kmer) {
                all_kmers.push(kmer.clone());
            }
        }

        let mut kl_pm = 0.0;
        let mut kl_qm = 0.0;
        let epsilon = 1e-10;

        for kmer in &all_kmers {
            let p = (self.counts.get(kmer).copied().unwrap_or(0) as f64 / self.total as f64)
                .max(epsilon);
            let q = (other.counts.get(kmer).copied().unwrap_or(0) as f64 / other.total as f64)
                .max(epsilon);
            let m = (p + q) / 2.0;

            if p > epsilon {
                kl_pm += p * (p / m).ln();
            }
            if q > epsilon {
                kl_qm += q * (q / m).ln();
            }
        }

        (kl_pm + kl_qm) / 2.0
    }
}

// =============================================================================
// KmerProfiler - High-level interface for sequence analysis
// =============================================================================

/// K-mer profiler for sequence analysis.
///
/// Maintains collections of sequences with their MinHash sketches and k-mer profiles
/// for efficient similarity comparisons.
#[cfg_attr(feature = "python", pyclass(subclass))]
#[derive(Clone, Debug)]
pub struct KmerProfiler {
    k: usize,
    normalize: bool,
    num_hashes: usize,
    sequences: AHashMap<String, Vec<u8>>,
    minhashes: AHashMap<String, MinHash>,
    profiles: AHashMap<String, KmerProfile>,
}

impl KmerProfiler {
    /// Create a new k-mer profiler.
    pub fn new(k: usize, normalize: bool, num_hashes: usize) -> Result<Self, KmerError> {
        if k < 1 {
            return Err(KmerError::InvalidK);
        }

        Ok(KmerProfiler {
            k,
            normalize,
            num_hashes,
            sequences: AHashMap::new(),
            minhashes: AHashMap::new(),
            profiles: AHashMap::new(),
        })
    }

    /// Add a sequence to the profiler.
    pub fn add_sequence(&mut self, seq_id: String, sequence: String) -> Result<(), KmerError> {
        if self.sequences.contains_key(&seq_id) {
            return Err(KmerError::DuplicateSequence(seq_id));
        }

        let cleaned_seq = clean_sequence(sequence.as_bytes())?;
        let seq_bytes = cleaned_seq.as_bytes().to_vec();

        let mut mh = MinHash::new(self.k, self.num_hashes);
        mh.add_sequence(&seq_bytes);

        let profile = KmerProfile::from_sequence(&seq_bytes, self.k);

        self.sequences.insert(seq_id.clone(), seq_bytes);
        self.minhashes.insert(seq_id.clone(), mh);
        self.profiles.insert(seq_id, profile);

        Ok(())
    }

    /// Calculate Jaccard similarity between two sequences.
    pub fn jaccard_similarity(&self, seq1: &str, seq2: &str) -> Result<f64, KmerError> {
        let mh1 = self
            .minhashes
            .get(seq1)
            .ok_or_else(|| KmerError::SequenceNotFound(seq1.to_string()))?;
        let mh2 = self
            .minhashes
            .get(seq2)
            .ok_or_else(|| KmerError::SequenceNotFound(seq2.to_string()))?;

        Ok(mh1.jaccard_similarity(mh2))
    }

    /// Calculate Jensen-Shannon divergence between two sequences.
    pub fn js_divergence(&self, seq1: &str, seq2: &str) -> Result<f64, KmerError> {
        let p1 = self
            .profiles
            .get(seq1)
            .ok_or_else(|| KmerError::SequenceNotFound(seq1.to_string()))?;
        let p2 = self
            .profiles
            .get(seq2)
            .ok_or_else(|| KmerError::SequenceNotFound(seq2.to_string()))?;

        Ok(p1.js_divergence(p2))
    }

    /// Get the raw sequence for a sequence ID.
    pub fn get_sequence(&self, seq_id: &str) -> Result<String, KmerError> {
        let seq = self
            .sequences
            .get(seq_id)
            .ok_or_else(|| KmerError::SequenceNotFound(seq_id.to_string()))?;
        Ok(String::from_utf8_lossy(seq).to_string())
    }

    /// Get list of all sequence IDs.
    pub fn sequence_ids(&self) -> Vec<String> {
        self.sequences.keys().cloned().collect()
    }

    /// Get k-mer counts for a sequence (Rust API).
    pub fn get_kmer_counts(&self, seq_id: &str) -> Result<AHashMap<String, f64>, KmerError> {
        let profile = self
            .profiles
            .get(seq_id)
            .ok_or_else(|| KmerError::SequenceNotFound(seq_id.to_string()))?;

        let mut result = AHashMap::new();
        for (kmer, count) in &profile.counts {
            let kmer_str = String::from_utf8_lossy(kmer).to_string();
            if self.normalize {
                result.insert(kmer_str, *count as f64 / profile.total as f64);
            } else {
                result.insert(kmer_str, *count as f64);
            }
        }
        Ok(result)
    }

    /// Get k value.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get number of sequences.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if profiler is empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Check if a sequence exists.
    pub fn contains(&self, seq_id: &str) -> bool {
        self.sequences.contains_key(seq_id)
    }
}

// Python-specific methods
#[cfg(feature = "python")]
#[pymethods]
impl KmerProfiler {
    #[new]
    #[pyo3(signature = (k=4, normalize=true, num_hashes=1000))]
    fn py_new(k: usize, normalize: bool, num_hashes: usize) -> PyResult<Self> {
        Self::new(k, normalize, num_hashes).map_err(|e| e.into())
    }

    #[pyo3(name = "add_sequence")]
    fn py_add_sequence(&mut self, seq_id: String, sequence: String) -> PyResult<()> {
        self.add_sequence(seq_id, sequence).map_err(|e| e.into())
    }

    #[pyo3(name = "jaccard_similarity")]
    fn py_jaccard_similarity(&self, seq1: &str, seq2: &str) -> PyResult<f64> {
        self.jaccard_similarity(seq1, seq2).map_err(|e| e.into())
    }

    #[pyo3(name = "js_divergence")]
    fn py_js_divergence(&self, seq1: &str, seq2: &str) -> PyResult<f64> {
        self.js_divergence(seq1, seq2).map_err(|e| e.into())
    }

    #[pyo3(name = "get_sequence")]
    fn py_get_sequence(&self, seq_id: &str) -> PyResult<String> {
        self.get_sequence(seq_id).map_err(|e| e.into())
    }

    #[pyo3(name = "sequence_ids")]
    fn py_sequence_ids(&self) -> Vec<String> {
        self.sequence_ids()
    }

    /// Get k-mer counts for a sequence as a Python dict.
    #[pyo3(name = "get_kmer_counts")]
    fn py_get_kmer_counts(&self, py: Python, seq_id: &str) -> PyResult<Py<PyDict>> {
        let profile = self
            .profiles
            .get(seq_id)
            .ok_or_else(|| KmerError::SequenceNotFound(seq_id.to_string()))?;

        let dict = PyDict::new(py);
        for (kmer, count) in &profile.counts {
            let kmer_str = String::from_utf8_lossy(kmer).to_string();
            if self.normalize {
                dict.set_item(kmer_str, *count as f64 / profile.total as f64)?;
            } else {
                dict.set_item(kmer_str, *count)?;
            }
        }
        Ok(dict.into_pyobject(py)?.into())
    }

    #[getter]
    fn get_k(&self) -> usize {
        self.k
    }

    fn __len__(&self) -> usize {
        self.sequences.len()
    }

    fn __contains__(&self, seq_id: &str) -> bool {
        self.sequences.contains_key(seq_id)
    }
}

// =============================================================================
// Python Functions
// =============================================================================

/// Build a k-mer profile from a sequence.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (sequence, k=4))]
pub fn build_kmer_profile(py: Python, sequence: &str, k: usize) -> PyResult<Py<PyDict>> {
    let cleaned = clean_sequence(sequence.as_bytes())?;
    let profile = KmerProfile::from_sequence(cleaned.as_bytes(), k);

    let dict = PyDict::new(py);
    for (kmer, count) in &profile.counts {
        let kmer_str = String::from_utf8_lossy(kmer).to_string();
        dict.set_item(kmer_str, *count as f64 / profile.total as f64)?;
    }
    Ok(dict.into_pyobject(py)?.into())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_hash() {
        let h1 = KmerHash::new(b"ATGC");
        let h2 = KmerHash::new(b"ATGC");
        let h3 = KmerHash::new(b"GCTA");
        assert_eq!(h1.hash, h2.hash);
        assert_ne!(h1.hash, h3.hash);
    }

    #[test]
    fn test_kmer_hash_canonical() {
        let h1 = KmerHash::canonical(b"ATG");
        let h2 = KmerHash::canonical(b"CAT");
        assert_eq!(h1.hash, h2.hash);
    }

    #[test]
    fn test_minhash_basic() {
        let mut mh = MinHash::new(3, 100);
        mh.add_sequence(b"ATGCATGCATGC");
        assert!(!mh.hashes.is_empty());
        assert!(mh.hashes.len() <= 100);
    }

    #[test]
    fn test_minhash_similarity_identical() {
        let mut mh1 = MinHash::new(3, 100);
        let mut mh2 = MinHash::new(3, 100);
        let seq = b"ATGCATGCATGCATGC";
        mh1.add_sequence(seq);
        mh2.add_sequence(seq);
        assert!((mh1.jaccard_similarity(&mh2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kmer_profile_basic() {
        let profile = KmerProfile::from_sequence(b"ATGCATGC", 2);
        assert!(profile.total > 0);
        assert!(!profile.counts.is_empty());
    }

    #[test]
    fn test_js_divergence_identical() {
        let p1 = KmerProfile::from_sequence(b"ATGCATGCATGC", 2);
        let p2 = KmerProfile::from_sequence(b"ATGCATGCATGC", 2);
        let div = p1.js_divergence(&p2);
        assert!(div < 0.01);
    }

    #[test]
    fn test_js_divergence_different() {
        let p1 = KmerProfile::from_sequence(b"AAAAAAAAAA", 2);
        let p2 = KmerProfile::from_sequence(b"GGGGGGGGGG", 2);
        let div = p1.js_divergence(&p2);
        assert!(div > 0.1);
    }

    #[test]
    fn test_kmer_profiler_basic() {
        let mut profiler = KmerProfiler::new(4, true, 1000).unwrap();
        assert!(profiler.add_sequence("seq1".to_string(), "ATGCATGCATGC".to_string()).is_ok());
        assert!(profiler.contains("seq1"));
        assert_eq!(profiler.len(), 1);
    }

    #[test]
    fn test_kmer_profiler_duplicate() {
        let mut profiler = KmerProfiler::new(4, true, 1000).unwrap();
        profiler.add_sequence("seq1".to_string(), "ATGC".to_string()).unwrap();
        assert!(profiler.add_sequence("seq1".to_string(), "GGGG".to_string()).is_err());
    }
}
