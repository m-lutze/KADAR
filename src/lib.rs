//! KADAR - K-mer based Anomaly Detection And Reporting
//!
//! This library provides tools for genomic island prediction based on k-mer
//! compositional analysis. Genomic islands are regions of DNA that have been
//! acquired through horizontal gene transfer and typically show different
//! sequence composition compared to the rest of the genome.
//!
//! # Architecture
//!
//! All functionality lives in the [`core`] module with conditional Python bindings.
//! When the "python" feature is enabled, types get PyO3 attributes for Python interop.
//!
//! - `core::seq` - DNA utilities and FASTA I/O
//! - `core::kmer` - K-mer hashing, MinHash, profiles
//! - `core::ivom` - IVOM algorithm (Vernikos & Parkhill 2006)
//! - `core::scanner` - Sliding window compositional scanning
//! - `core::predictor` - Main genome island prediction
//! - `core::types` - Data structures (GenomicIsland, WindowScore)
//!
//! # Example (Python)
//!
//! ```python
//! from kadar import GenomeIslandPredictor, load_fasta
//!
//! # Load sequences from FASTA
//! sequences = load_fasta("genome.fasta")
//!
//! # Create predictor
//! predictor = GenomeIslandPredictor(k=4, window_size=5000)
//!
//! # Predict islands
//! for seq_id, sequence in sequences:
//!     islands = predictor.predict(sequence, seq_id)
//!     for island in islands:
//!         print(f"{island.sequence_id}: {island.start}-{island.end}")
//! ```
//!
//! # Example (Rust)
//!
//! ```rust,ignore
//! use kadar::{GenomeIslandPredictor, KmerProfile};
//!
//! let mut predictor = GenomeIslandPredictor::new(4, 5000, 1000, 8000, 0.1)?;
//! predictor.fit(&genome_sequence)?;
//! let islands = predictor.predict(&sequence, Some("seq1"))?;
//! ```

#[cfg(feature = "python")]
use pyo3::prelude::*;

// Core module - all implementations live here
pub mod core;

// Re-exports for Rust API convenience
pub use core::{
    canonical_kmer, clean_sequence, gc_content, reverse_complement,
    GenomeIslandPredictor, GenomicIsland, IVOMAnalysis, IvomModel,
    KmerHash, KmerProfile, KmerProfiler, MinHash,
    SlidingWindowScanner, WindowScore,
    IvomError, KmerError, PredictorError, ScannerError, SeqError,
    relative_entropy,
};

#[cfg(feature = "fasta")]
pub use core::{load_fasta_file, load_fasta_str, FastaRecord};

// =============================================================================
// Python Module Registration
// =============================================================================

#[cfg(feature = "python")]
#[pymodule]
fn kadar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<core::GenomicIsland>()?;
    m.add_class::<core::WindowScore>()?;
    m.add_class::<core::KmerProfiler>()?;
    m.add_class::<core::IVOMAnalysis>()?;
    m.add_class::<core::SlidingWindowScanner>()?;
    m.add_class::<core::GenomeIslandPredictor>()?;

    // Functions
    m.add_function(wrap_pyfunction!(core::py_gc_content, m)?)?;
    m.add_function(wrap_pyfunction!(core::build_kmer_profile, m)?)?;

    #[cfg(feature = "fasta")]
    {
        m.add_function(wrap_pyfunction!(core::load_fasta, m)?)?;
        m.add_function(wrap_pyfunction!(core::load_fasta_string, m)?)?;
    }

    Ok(())
}
