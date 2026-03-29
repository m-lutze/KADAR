//! DNA sequence utilities and FASTA I/O.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use thiserror::Error;

// =============================================================================
// Errors
// =============================================================================

#[derive(Error, Debug)]
pub enum SeqError {
    #[error("Sequence is empty after cleaning")]
    EmptySequence,
    #[error("UTF-8 error")]
    Utf8Error,
    #[error("FASTA error: {0}")]
    FastaError(String),
}

#[cfg(feature = "python")]
impl From<SeqError> for PyErr {
    fn from(err: SeqError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// =============================================================================
// DNA Sequence Utilities
// =============================================================================

/// Clean DNA sequence - uppercase, remove invalid chars.
pub fn clean_sequence(seq: &[u8]) -> Result<String, SeqError> {
    let mut cleaned = Vec::with_capacity(seq.len());

    for &byte in seq {
        match byte.to_ascii_uppercase() {
            b'A' | b'T' | b'G' | b'C' => cleaned.push(byte.to_ascii_uppercase()),
            b'N' | b'R' | b'Y' | b'S' | b'W' | b'K' | b'M' | b'B' | b'D' | b'H' | b'V' => {}
            b' ' | b'\n' | b'\r' | b'\t' => {}
            _ => {}
        }
    }

    if cleaned.is_empty() {
        return Err(SeqError::EmptySequence);
    }

    String::from_utf8(cleaned).map_err(|_| SeqError::Utf8Error)
}

/// Get reverse complement of a DNA sequence.
#[inline]
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' => b'T',
            b'T' => b'A',
            b'G' => b'C',
            b'C' => b'G',
            _ => b,
        })
        .collect()
}

/// Get canonical k-mer (lexicographically smaller of forward/reverse complement).
#[inline]
pub fn canonical_kmer(kmer: &[u8]) -> Vec<u8> {
    let rc = reverse_complement(kmer);
    if kmer <= &rc[..] {
        kmer.to_vec()
    } else {
        rc
    }
}

/// Calculate GC content of a sequence.
#[inline]
pub fn gc_content(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let gc_count = seq.iter().filter(|&&b| b == b'G' || b == b'C').count();
    gc_count as f64 / seq.len() as f64
}

// =============================================================================
// Python Functions
// =============================================================================

/// Calculate GC content (Python-accessible).
#[cfg(feature = "python")]
#[pyfunction]
pub fn py_gc_content(sequence: &str) -> PyResult<f64> {
    let cleaned = clean_sequence(sequence.as_bytes())?;
    Ok(gc_content(cleaned.as_bytes()))
}

// =============================================================================
// FASTA I/O
// =============================================================================

/// A parsed FASTA record.
#[derive(Clone, Debug)]
pub struct FastaRecord {
    pub id: String,
    pub sequence: String,
}

#[cfg(feature = "fasta")]
pub fn load_fasta_file(path: &str) -> Result<Vec<FastaRecord>, SeqError> {
    use needletail::parse_fastx_file;

    let mut reader = parse_fastx_file(path)
        .map_err(|e| SeqError::FastaError(format!("Failed to open: {}", e)))?;

    let mut records = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.map_err(|e| SeqError::FastaError(e.to_string()))?;
        records.push(FastaRecord {
            id: String::from_utf8_lossy(record.id()).to_string(),
            sequence: String::from_utf8_lossy(&record.seq()).to_string(),
        });
    }
    Ok(records)
}

#[cfg(feature = "fasta")]
pub fn load_fasta_str(content: &str) -> Result<Vec<FastaRecord>, SeqError> {
    use needletail::parse_fastx_reader;

    let cursor = std::io::Cursor::new(content.as_bytes());
    let mut reader = parse_fastx_reader(cursor)
        .map_err(|e| SeqError::FastaError(format!("Failed to parse: {}", e)))?;

    let mut records = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.map_err(|e| SeqError::FastaError(e.to_string()))?;
        records.push(FastaRecord {
            id: String::from_utf8_lossy(record.id()).to_string(),
            sequence: String::from_utf8_lossy(&record.seq()).to_string(),
        });
    }
    Ok(records)
}

/// Load sequences from a FASTA file (Python-accessible).
#[cfg(all(feature = "python", feature = "fasta"))]
#[pyfunction]
pub fn load_fasta(path: &str) -> PyResult<Vec<(String, String)>> {
    let records = load_fasta_file(path)?;
    Ok(records.into_iter().map(|r| (r.id, r.sequence)).collect())
}

/// Load sequences from FASTA string (Python-accessible).
#[cfg(all(feature = "python", feature = "fasta"))]
#[pyfunction]
pub fn load_fasta_string(content: &str) -> PyResult<Vec<(String, String)>> {
    let records = load_fasta_str(content)?;
    Ok(records.into_iter().map(|r| (r.id, r.sequence)).collect())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_complement() {
        assert_eq!(reverse_complement(b"ATGC"), b"GCAT");
        assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
    }

    #[test]
    fn test_canonical_kmer() {
        assert_eq!(canonical_kmer(b"ATG"), b"ATG".to_vec());
        assert_eq!(canonical_kmer(b"CAT"), b"ATG".to_vec());
    }

    #[test]
    fn test_gc_content() {
        assert_eq!(gc_content(b"GGCC"), 1.0);
        assert_eq!(gc_content(b"AATT"), 0.0);
        assert_eq!(gc_content(b"ATGC"), 0.5);
    }

    #[test]
    fn test_clean_sequence() {
        assert_eq!(clean_sequence(b"atgc").unwrap(), "ATGC");
        assert!(clean_sequence(b"NNNN").is_err());
    }

    #[cfg(feature = "fasta")]
    #[test]
    fn test_load_fasta_str() {
        let fasta = ">seq1\nATGC\n>seq2\nGGGG\n";
        let records = load_fasta_str(fasta).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "seq1");
    }
}
