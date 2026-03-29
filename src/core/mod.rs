//! Core algorithms and types.
//!
//! This module contains all the core implementations with conditional Python bindings.
//! When the "python" feature is enabled, types get PyO3 attributes for Python interop.

pub mod ivom;
pub mod kmer;
pub mod predictor;
pub mod scanner;
pub mod seq;
pub mod types;

// Re-export common types
pub use ivom::{relative_entropy, IVOMAnalysis, IvomError, IvomModel};
pub use kmer::{KmerError, KmerHash, KmerProfile, KmerProfiler, MinHash};
pub use predictor::{GenomeIslandPredictor, PredictorError};
pub use scanner::{ScannerError, SlidingWindowScanner};
pub use seq::{canonical_kmer, clean_sequence, gc_content, reverse_complement, SeqError};
pub use types::{GenomicIsland, WindowScore};

// FASTA functionality when enabled
#[cfg(feature = "fasta")]
pub use seq::{load_fasta_file, load_fasta_str, FastaRecord};

// Re-export Python functions for module registration, TD: clean this 
#[cfg(feature = "python")]
pub use kmer::build_kmer_profile;
#[cfg(feature = "python")]
pub use seq::py_gc_content;
#[cfg(all(feature = "python", feature = "fasta"))]
pub use seq::{load_fasta, load_fasta_string};
