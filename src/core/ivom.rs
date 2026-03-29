//! IVOM (Interpolated Variable Order Motifs) implementation.
//!
//! Ref: Vernikos & Parkhill (2006) "Interpolated variable order motifs for
//! identification of horizontally acquired DNA: revisiting the Salmonella
//! pathogenicity islands" Bioinformatics 22(18):2196-2203
//!
//! The IVOM approach uses variable order k-mers (1 to max_order, typically 8),
//! preferring information from high order motifs when reliable, but falling back
//! to lower order motifs when counts are insufficient.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use ahash::AHashMap;
use thiserror::Error;

use super::seq::{clean_sequence, SeqError};

// =============================================================================
// Errors
// =============================================================================

#[derive(Error, Debug)]
pub enum IvomError {
    #[error("max_order must be >= min_order")]
    InvalidOrderRange,
    #[error("min_order must be >= 1")]
    InvalidMinOrder,
    #[error("max_order > 15 is not recommended (memory/computation)")]
    OrderTooHigh,
    #[error("Background model not built. Call build_background_model first.")]
    ModelNotBuilt,
    #[error("Sequence error: {0}")]
    SeqError(#[from] SeqError),
}

#[cfg(feature = "python")]
impl From<IvomError> for PyErr {
    fn from(err: IvomError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// =============================================================================
// IvomModel - Core implementation
// =============================================================================

/// IVOM frequency calculator for interpolation scheme.
///
/// For each k-mer m in sequence S:
/// - Observed frequency: P_m(S) = A_m(S) / N
/// - Weight: W_m(S) = A_m(S) / |B|^k        
/// - IVOM frequency: recursive interpolation 
#[derive(Clone, Debug)]
pub struct IvomModel {
    /// Maximum k-mer order (paper uses 8)
    pub max_order: usize,
    /// Minimum k-mer order (paper uses 1)
    pub min_order: usize,
    /// K-mer counts for each order: counts[order-min_order][kmer] = count
    counts: Vec<AHashMap<Vec<u8>, u64>>,
    /// Total k-mers counted at each order
    totals: Vec<u64>,
    /// Sequence length used for building model
    seq_length: usize,
}

impl IvomModel {
    /// Create a new IVOM model with specified order range.
    ///
    /// Paper recommends max_order=8, min_order=1.
    pub fn new(min_order: usize, max_order: usize) -> Self {
        assert!(min_order >= 1, "min_order must be >= 1");
        assert!(max_order >= min_order, "max_order must be >= min_order");

        let n_orders = max_order - min_order + 1;
        let counts = (0..n_orders).map(|_| AHashMap::new()).collect();
        let totals = vec![0; n_orders];

        IvomModel {
            max_order,
            min_order,
            counts,
            totals,
            seq_length: 0,
        }
    }

    /// Build model from a DNA sequence.
    ///
    /// Counts all k-mers of orders min_order to max_order.
    pub fn build(&mut self, seq: &[u8]) {
        self.seq_length = seq.len();

        // Clear existing counts
        for counts in &mut self.counts {
            counts.clear();
        }
        for total in &mut self.totals {
            *total = 0;
        }

        // Count k-mers at each order
        for order in self.min_order..=self.max_order {
            let idx = order - self.min_order;
            if seq.len() >= order {
                for i in 0..=(seq.len() - order) {
                    let kmer = &seq[i..i + order];
                    // Only count valid DNA k-mers
                    if kmer.iter().all(|&b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                        *self.counts[idx].entry(kmer.to_vec()).or_insert(0) += 1;
                        self.totals[idx] += 1;
                    }
                }
            }
        }
    }

    /// Build model from multiple sequences (e.g., full genome).
    pub fn build_from_sequences(&mut self, sequences: &[&[u8]]) {
        // Clear existing counts
        for counts in &mut self.counts {
            counts.clear();
        }
        for total in &mut self.totals {
            *total = 0;
        }
        self.seq_length = 0;

        for seq in sequences {
            self.seq_length += seq.len();

            for order in self.min_order..=self.max_order {
                let idx = order - self.min_order;
                if seq.len() >= order {
                    for i in 0..=(seq.len() - order) {
                        let kmer = &seq[i..i + order];
                        if kmer.iter().all(|&b| matches!(b, b'A' | b'T' | b'G' | b'C')) {
                            *self.counts[idx].entry(kmer.to_vec()).or_insert(0) += 1;
                            self.totals[idx] += 1;
                        }
                    }
                }
            }
        }
    }

    /// Get observed frequency P_m(S) for a k-mer.
    ///
    /// P_m(S) = A_m(S) / N
    /// where A_m(S) is the count of m in S, and N is the sequence length.
    #[inline]
    pub fn observed_frequency(&self, kmer: &[u8]) -> f64 {
        let order = kmer.len();
        if order < self.min_order || order > self.max_order {
            return 0.0;
        }
        let idx = order - self.min_order;
        let count = self.counts[idx].get(kmer).copied().unwrap_or(0);

        if self.seq_length == 0 {
            0.0
        } else {
            count as f64 / self.seq_length as f64
        }
    }

    /// Get weight W_m(S) for a k-mer.
    ///
    /// W_m(S) = A_m(S) / |B|^k
    /// where |B| = 4 (DNA alphabet size) and k is the k-mer length.
    ///
    /// This normalizes counts by the total possible k-mers of that order,
    /// giving high and low order motifs equal chances of producing bias.
    #[inline]
    pub fn weight(&self, kmer: &[u8]) -> f64 {
        let order = kmer.len();
        if order < self.min_order || order > self.max_order {
            return 0.0;
        }
        let idx = order - self.min_order;
        let count = self.counts[idx].get(kmer).copied().unwrap_or(0);

        // |B|^k = 4^k
        let alphabet_power = 4_u64.pow(order as u32);
        count as f64 / alphabet_power as f64
    }

    /// Get IVOM frequency for a k-mer.
    ///
    /// IVOM_m(S) = W_m(S) * P_m(S) + (1 - W_m(S)) * IVOM_{m[2:|m|]}(S)
    ///
    /// This recursively combines observed frequencies: if high order motifs
    /// have sufficient counts (high weight), their frequency dominates;
    /// otherwise, lower order motif frequencies contribute more.
    pub fn ivom_frequency(&self, kmer: &[u8]) -> f64 {
        if kmer.is_empty() {
            return 0.0;
        }

        let order = kmer.len();

        // Base case: 1-mer or below min_order
        if order <= self.min_order {
            return self.observed_frequency(kmer);
        }

        // If order is above max_order, use substring
        if order > self.max_order {
            return self.ivom_frequency(&kmer[1..]);
        }

        let w = self.weight(kmer);
        let p = self.observed_frequency(kmer);

        // Recursive case: interpolate with suffix (m[2:|m|] = kmer[1..])
        // The suffix is the k-mer starting at position 2 (1-indexed in paper)
        let suffix = &kmer[1..];
        let ivom_suffix = self.ivom_frequency(suffix);

        // Equation 3: IVOM_m = W_m * P_m + (1 - W_m) * IVOM_{suffix}
        w * p + (1.0 - w) * ivom_suffix
    }

    /// Build IVOM compositional vector for all 8-mers.
    ///
    /// Returns a vector of IVOM frequencies indexed by k-mer encoding.
    /// This extends over all |B|^8 = 65536 possible 8-mers.
    pub fn build_ivom_vector(&self) -> Vec<f64> {
        let num_kmers = 4_usize.pow(self.max_order as u32);
        let mut vector = Vec::with_capacity(num_kmers);

        // Generate all possible max_order k-mers and compute IVOM frequency
        let mut kmer = vec![b'A'; self.max_order];
        for _ in 0..num_kmers {
            vector.push(self.ivom_frequency(&kmer));
            // Increment k-mer (base-4 counter: A=0, C=1, G=2, T=3)
            increment_kmer(&mut kmer);
        }

        vector
    }

    /// Get raw count for a specific k-mer.
    pub fn get_count(&self, kmer: &[u8]) -> u64 {
        let order = kmer.len();
        if order < self.min_order || order > self.max_order {
            return 0;
        }
        let idx = order - self.min_order;
        self.counts[idx].get(kmer).copied().unwrap_or(0)
    }
}

// =============================================================================
// IVOMAnalysis - High-level interface for genome island detection
// =============================================================================

/// IVOM (Interpolated Variable Order Motifs) analysis for genome island detection.
///
/// Implements algorithm largely from Vernikos & Parkhill (2006):
/// - Uses variable order k-mers from min_order to max_order
/// - Weights each k-mer by its count relative to total possible k-mers of that order
/// - Interpolates frequencies: high order motifs are preferred when reliable,
///   falling back to lower order when counts are insufficient
/// - Compares windows to genome background using relative entropy (KL divergence)
#[cfg_attr(feature = "python", pyclass(subclass))]
#[derive(Clone)]
pub struct IVOMAnalysis {
    /// Maximum k-mer order (paper recommends 8)
    pub max_order: usize,
    /// Minimum k-mer order (paper uses 1)
    pub min_order: usize,
    /// Background model built from genome
    background_model: Option<IvomModel>,
    /// Precomputed background IVOM vector for efficiency
    background_vector: Option<Vec<f64>>,
}

impl IVOMAnalysis {
    /// Create a new IVOM analyzer.
    pub fn new(max_order: usize, min_order: usize) -> Result<Self, IvomError> {
        if max_order < min_order {
            return Err(IvomError::InvalidOrderRange);
        }
        if min_order < 1 {
            return Err(IvomError::InvalidMinOrder);
        }
        if max_order > 10 {
            return Err(IvomError::OrderTooHigh);
        }

        Ok(IVOMAnalysis {
            max_order,
            min_order,
            background_model: None,
            background_vector: None,
        })
    }

    /// Build background model from reference sequences (typically the full genome).
    pub fn build_background_model(&mut self, sequences: Vec<String>) -> Result<(), IvomError> {
        let cleaned: Vec<Vec<u8>> = sequences
            .iter()
            .map(|s| clean_sequence(s.as_bytes()).map(|c| c.into_bytes()))
            .collect::<Result<Vec<_>, _>>()?;

        let refs: Vec<&[u8]> = cleaned.iter().map(|v| v.as_slice()).collect();

        let mut model = IvomModel::new(self.min_order, self.max_order);
        model.build_from_sequences(&refs);

        // Precompute the background IVOM vector for efficiency
        let vector = model.build_ivom_vector();

        self.background_model = Some(model);
        self.background_vector = Some(vector);

        Ok(())
    }

    /// Calculate compositional deviation score for a sequence.
    pub fn calculate_deviation(&self, sequence: &str) -> Result<f64, IvomError> {
        let bg_vector = self
            .background_vector
            .as_ref()
            .ok_or(IvomError::ModelNotBuilt)?;

        let cleaned = clean_sequence(sequence.as_bytes())?;
        let seq_bytes = cleaned.as_bytes();

        // Build IVOM model for this window
        let mut window_model = IvomModel::new(self.min_order, self.max_order);
        window_model.build(seq_bytes);

        // Compute IVOM vector for window
        let window_vector = window_model.build_ivom_vector();

        // Calculate relative entropy (KL divergence)
        let entropy = relative_entropy(&window_vector, bg_vector);

        Ok(entropy)
    }

    /// Calculate IVOM frequency for a specific k-mer against the background.
    pub fn get_background_ivom_frequency(&self, kmer: &str) -> Result<f64, IvomError> {
        let model = self
            .background_model
            .as_ref()
            .ok_or(IvomError::ModelNotBuilt)?;

        let cleaned = clean_sequence(kmer.as_bytes())?;
        Ok(model.ivom_frequency(cleaned.as_bytes()))
    }

    /// Get the weight for a k-mer in the background model.
    pub fn get_kmer_weight(&self, kmer: &str) -> Result<f64, IvomError> {
        let model = self
            .background_model
            .as_ref()
            .ok_or(IvomError::ModelNotBuilt)?;

        let cleaned = clean_sequence(kmer.as_bytes())?;
        Ok(model.weight(cleaned.as_bytes()))
    }

    /// Get the observed frequency for a k-mer in the background model.
    pub fn get_observed_frequency(&self, kmer: &str) -> Result<f64, IvomError> {
        let model = self
            .background_model
            .as_ref()
            .ok_or(IvomError::ModelNotBuilt)?;

        let cleaned = clean_sequence(kmer.as_bytes())?;
        Ok(model.observed_frequency(cleaned.as_bytes()))
    }

    /// Get the interpolation weights for visualization.
    pub fn get_weights(&self) -> Vec<f64> {
        (self.min_order..=self.max_order)
            .map(|k| 1.0 / (4.0_f64.powi(k as i32)))
            .collect()
    }

    /// Check if background model has been built.
    pub fn is_fitted(&self) -> bool {
        self.background_model.is_some()
    }
}

// Python-specific methods
#[cfg(feature = "python")]
#[pymethods]
impl IVOMAnalysis {
    #[new]
    #[pyo3(signature = (max_order=8, min_order=1))]
    fn py_new(max_order: usize, min_order: usize) -> PyResult<Self> {
        Self::new(max_order, min_order).map_err(|e| e.into())
    }

    #[pyo3(name = "build_background_model")]
    fn py_build_background_model(&mut self, sequences: Vec<String>) -> PyResult<()> {
        self.build_background_model(sequences).map_err(|e| e.into())
    }

    #[pyo3(name = "calculate_deviation")]
    fn py_calculate_deviation(&self, sequence: &str) -> PyResult<f64> {
        self.calculate_deviation(sequence).map_err(|e| e.into())
    }

    #[pyo3(name = "get_background_ivom_frequency")]
    fn py_get_background_ivom_frequency(&self, kmer: &str) -> PyResult<f64> {
        self.get_background_ivom_frequency(kmer).map_err(|e| e.into())
    }

    #[pyo3(name = "get_kmer_weight")]
    fn py_get_kmer_weight(&self, kmer: &str) -> PyResult<f64> {
        self.get_kmer_weight(kmer).map_err(|e| e.into())
    }

    #[pyo3(name = "get_observed_frequency")]
    fn py_get_observed_frequency(&self, kmer: &str) -> PyResult<f64> {
        self.get_observed_frequency(kmer).map_err(|e| e.into())
    }

    #[pyo3(name = "get_weights")]
    fn py_get_weights(&self) -> Vec<f64> {
        self.get_weights()
    }

    #[getter]
    fn get_max_order(&self) -> usize {
        self.max_order
    }

    #[getter]
    fn get_min_order(&self) -> usize {
        self.min_order
    }

    #[pyo3(name = "is_fitted")]
    fn py_is_fitted(&self) -> bool {
        self.is_fitted()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate relative entropy (KL divergence) between window and genome.
///
/// D(w || G) = Σ_m IVOM_m(w) * log(IVOM_m(w) / IVOM_m(G))
///
/// Regions of atypical composition will have high relative entropy,
/// while native regions will have entropy close to zero.
pub fn relative_entropy(window_ivom: &[f64], genome_ivom: &[f64]) -> f64 {
    assert_eq!(
        window_ivom.len(),
        genome_ivom.len(),
        "IVOM vectors must have same length"
    );

    let epsilon = 1e-10; // Avoid log(0)
    let mut entropy = 0.0;

    for (w, g) in window_ivom.iter().zip(genome_ivom.iter()) {
        let w_freq = w.max(epsilon);
        let g_freq = g.max(epsilon);

        // Only add contribution if window frequency is non-negligible
        if *w > epsilon {
            entropy += w_freq * (w_freq / g_freq).ln();
        }
    }

    entropy
}

/// Calculate symmetric KL divergence (Jensen-Shannon style but using KL).
///
/// Better than asymmetric KL when either distribution may have zeros.
pub fn symmetric_relative_entropy(ivom1: &[f64], ivom2: &[f64]) -> f64 {
    (relative_entropy(ivom1, ivom2) + relative_entropy(ivom2, ivom1)) / 2.0
}

/// Increment a k-mer like a base-4 counter (A < C < G < T).
fn increment_kmer(kmer: &mut [u8]) {
    for base in kmer.iter_mut().rev() {
        match *base {
            b'A' => {
                *base = b'C';
                return;
            }
            b'C' => {
                *base = b'G';
                return;
            }
            b'G' => {
                *base = b'T';
                return;
            }
            b'T' => {
                *base = b'A';
                // Continue to next position (carry)
            }
            _ => return,
        }
    }
}

/// Encode a k-mer as an integer index (for vector indexing).
pub fn encode_kmer(kmer: &[u8]) -> usize {
    let mut index = 0;
    for &base in kmer {
        index *= 4;
        index += match base {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        };
    }
    index
}

/// Decode an integer index back to a k-mer.
pub fn decode_kmer(mut index: usize, k: usize) -> Vec<u8> {
    let mut kmer = vec![b'A'; k];
    for i in (0..k).rev() {
        kmer[i] = match index % 4 {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            3 => b'T',
            _ => b'A',
        };
        index /= 4;
    }
    kmer
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivom_model_basic() {
        let mut model = IvomModel::new(1, 4);
        model.build(b"ATGCATGCATGC");

        // Check that counts exist
        assert!(model.get_count(b"A") > 0);
        assert!(model.get_count(b"AT") > 0);
        assert!(model.get_count(b"ATG") > 0);
    }

    #[test]
    fn test_observed_frequency() {
        let mut model = IvomModel::new(1, 4);
        model.build(b"AAAA"); // 4 A's, seq_length = 4

        let freq = model.observed_frequency(b"A");
        // A appears 4 times in sequence of length 4
        assert!((freq - 1.0).abs() < 0.01, "Expected ~1.0, got {}", freq);
    }

    #[test]
    fn test_weight() {
        let mut model = IvomModel::new(1, 4);
        model.build(b"ATGCATGCATGCATGC"); // 16 bp

        // Weight for 1-mer: count / 4^1 = count / 4
        // A appears 4 times, so weight = 4/4 = 1.0
        let w = model.weight(b"A");
        assert!((w - 1.0).abs() < 0.01, "1-mer weight wrong: {}", w);

        // Weight for 2-mer: count / 4^2 = count / 16
        // AT appears 4 times, so weight = 4/16 = 0.25
        let w2 = model.weight(b"AT");
        assert!((w2 - 0.25).abs() < 0.01, "2-mer weight wrong: {}", w2);
    }

    #[test]
    fn test_ivom_frequency_base_case() {
        let mut model = IvomModel::new(1, 4);
        model.build(b"AAAA");

        // For 1-mers (base case), IVOM frequency equals observed frequency
        let ivom = model.ivom_frequency(b"A");
        let obs = model.observed_frequency(b"A");
        assert!(
            (ivom - obs).abs() < 0.001,
            "Base case: IVOM {} != observed {}",
            ivom,
            obs
        );
    }

    #[test]
    fn test_ivom_frequency_interpolation() {
        let mut model = IvomModel::new(1, 4);
        // Sequence with varied composition
        model.build(b"ATGCATGCATGCATGCATGCATGCATGCATGC");

        // Higher order k-mers should have interpolated frequencies
        let ivom_4mer = model.ivom_frequency(b"ATGC");
        assert!(ivom_4mer > 0.0, "4-mer IVOM should be > 0");
        assert!(ivom_4mer < 1.0, "4-mer IVOM should be < 1");
    }

    #[test]
    fn test_relative_entropy_identical() {
        let vec1 = vec![0.25, 0.25, 0.25, 0.25];
        let vec2 = vec![0.25, 0.25, 0.25, 0.25];

        let entropy = relative_entropy(&vec1, &vec2);
        assert!(
            entropy.abs() < 0.001,
            "Identical distributions should have ~0 entropy, got {}",
            entropy
        );
    }

    #[test]
    fn test_relative_entropy_different() {
        let vec1 = vec![0.7, 0.1, 0.1, 0.1];
        let vec2 = vec![0.25, 0.25, 0.25, 0.25];

        let entropy = relative_entropy(&vec1, &vec2);
        assert!(
            entropy > 0.1,
            "Different distributions should have positive entropy, got {}",
            entropy
        );
    }

    #[test]
    fn test_encode_decode_kmer() {
        let kmer = b"ATGC";
        let encoded = encode_kmer(kmer);
        let decoded = decode_kmer(encoded, 4);
        assert_eq!(&decoded, kmer);
    }

    #[test]
    fn test_increment_kmer() {
        let mut kmer = vec![b'A', b'A'];
        increment_kmer(&mut kmer);
        assert_eq!(kmer, vec![b'A', b'C']);

        let mut kmer2 = vec![b'A', b'T'];
        increment_kmer(&mut kmer2);
        assert_eq!(kmer2, vec![b'C', b'A']);
    }

    #[test]
    fn test_ivom_analysis_creation() {
        let ivom = IVOMAnalysis::new(8, 1).unwrap();
        assert_eq!(ivom.max_order, 8);
        assert_eq!(ivom.min_order, 1);
        assert!(!ivom.is_fitted());
    }

    #[test]
    fn test_ivom_invalid_params() {
        // max < min should fail
        assert!(IVOMAnalysis::new(2, 4).is_err());
        // min < 1 should fail
        assert!(IVOMAnalysis::new(4, 0).is_err());
    }

    #[test]
    fn test_ivom_build_model() {
        let mut ivom = IVOMAnalysis::new(4, 1).unwrap();
        let result = ivom.build_background_model(vec!["ATGCATGCATGCATGC".to_string()]);
        assert!(result.is_ok());
        assert!(ivom.is_fitted());
    }

    #[test]
    fn test_ivom_deviation_without_model() {
        let ivom = IVOMAnalysis::new(4, 1).unwrap();
        let result = ivom.calculate_deviation("ATGC");
        assert!(result.is_err()); // Should fail without background model
    }
}
