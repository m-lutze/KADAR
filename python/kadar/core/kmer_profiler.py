import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

# sourmash is required
import sourmash
from sourmash import MinHash
from sourmash.signature import SourmashSignature


# sourmash utility functions
def extract_kmers_from_sig(signature):
    """extract k-mer hashes from sourmash signature"""
    return list(signature.minhash.hashes)


def clean_sequence(seq):
    """clean up sequence - make uppercase and check for weird characters"""
    if not seq:
        raise ValueError("empty sequence")

    seq = seq.upper().strip()
    # basic check for valid DNA bases - sourmash doesn't like N chars
    valid = set('ATGC')
    bad_chars = set(seq) - valid - {'N'}  # Allow N but we'll remove them
    if bad_chars:
        raise ValueError(f"Found invalid characters: {bad_chars}")

    # Remove N characters since sourmash can't handle them
    seq = seq.replace('N', '')

    if not seq:
        raise ValueError("Sequence contains only N characters or is empty after cleaning")

    return seq


class KmerProfiler:
    """
    Main class for k-mer stuff.

    Handles k-mer profiles, sourmash integration, and basic analysis.
    Keeps track of sequences and their k-mer composition.
    """
    
    def __init__(self, k=4, normalize=True, num_hashes=1000, scaled=1000):
        """
        Set up the profiler

        Args:
            k: k-mer size (default 4)
            normalize: whether to normalize frequencies
            num_hashes: number of hashes for sourmash
            scaled: scaled parameter for sourmash
        """
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.normalize = normalize
        self.num_hashes = num_hashes
        self.scaled = scaled

        # data storage - use sourmash as primary storage
        # may be able to optimise this further
        self.sequences = {}
        self.sourmash_signatures = {}
        self.sequence_metadata = {}
        
    def add_sequence(self, seq_id, sequence, metadata=None):
        """
        Add a sequence to the profiler

        Args:
            seq_id: sequence identifier (must be unique)
            sequence: DNA sequence string
            metadata: optional dict with extra info
        """
        if seq_id in self.sequences:
            raise ValueError(f"Sequence {seq_id} already exists!")

        # clean up the sequence
        sequence = clean_sequence(sequence)

        self.sequences[seq_id] = sequence
        if metadata:
            self.sequence_metadata[seq_id] = metadata

        # make sourmash signature - this is our primary k-mer storage
        self.sourmash_signatures[seq_id] = self.make_sourmash_sig(seq_id, sequence)
        
    def make_sourmash_sig(self, seq_id, sequence):
        """make a sourmash signature for the sequence"""
        # set up minhash
        if self.scaled > 0:
            mh = MinHash(n=0, ksize=self.k, scaled=self.scaled)
        else:
            mh = MinHash(n=self.num_hashes, ksize=self.k)

        mh.add_sequence(sequence)
        sig = SourmashSignature(mh, name=seq_id)
        return sig
    
    def get_profile_matrix(self, seq_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get k-mer profile matrix for specified sequences using sourmash hashes.
        """
        if seq_ids is None:
            seq_ids = list(self.sequences.keys())

        # Validate sequence IDs
        missing_ids = [seq_id for seq_id in seq_ids if seq_id not in self.sequences]
        if missing_ids:
            raise ValueError(f"Sequence IDs not found: {missing_ids}")

        # Get the hash matrix (this should work with sourmash)
        matrix, valid_seq_ids, hashes = self.get_hash_matrix(seq_ids)

        # Convert hashes to string representations for compatibility
        hash_list = [str(h) for h in hashes]

        return matrix, valid_seq_ids, hash_list
    
    def get_kmer_hashes(self, seq_id):
        """get k-mer hashes from sourmash signature"""
        sig = self.get_signature(seq_id)
        return extract_kmers_from_sig(sig)
    
    def get_signature(self, seq_id):
        """get sourmash signature for a sequence"""
        if seq_id not in self.sourmash_signatures:
            raise KeyError(f"No signature for {seq_id}")
        return self.sourmash_signatures[seq_id]

    def jaccard_similarity(self, seq1, seq2):
        """calculate jaccard similarity between two sequences"""
        sig1 = self.get_signature(seq1)
        sig2 = self.get_signature(seq2)
        return sig1.similarity(sig2)

    def containment(self, query_seq, subject_seq):
        """how much of query is contained in subject (0-1)"""
        sig1 = self.get_signature(query_seq)
        sig2 = self.get_signature(subject_seq)
        return sig1.contained_by(sig2)

    def get_hash_vector(self, seq_id):
        """get the minhash vector as numpy array"""
        sig = self.get_signature(seq_id)
        return np.array(sig.minhash.hashes)

    def get_hash_matrix(self, seq_ids=None):
        """
        Get sourmash hash matrix for sequences

        Returns matrix where each row is a sequence and columns are presence/absence of hashes
        """
        if not self.sequences:
            raise ValueError("no sequences to work with")

        if seq_ids is None:
            seq_ids = list(self.sequences.keys())

        # check if sequences exist
        missing = [s for s in seq_ids if s not in self.sequences]
        if missing:
            raise ValueError(f"sequences not found: {missing}")

        # collect all unique hashes across all signatures
        all_hashes = set()
        for seq_id in seq_ids:
            hashes = self.get_kmer_hashes(seq_id)
            all_hashes.update(hashes)

        hash_list = sorted(all_hashes)
        matrix = np.zeros((len(seq_ids), len(hash_list)))

        # fill matrix with presence/absence of hashes
        for i, seq_id in enumerate(seq_ids):
            seq_hashes = set(self.get_kmer_hashes(seq_id))
            for j, hash_val in enumerate(hash_list):
                matrix[i, j] = 1 if hash_val in seq_hashes else 0

        return matrix, seq_ids, hash_list

    def get_sourmash_similarity_matrix(self, seq_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Get sourmash-based similarity matrix for specified sequences.

        Args:
            seq_ids: List of sequence IDs to include (all if None)

        Returns:
            Tuple of (similarity_matrix, sequence_ids)
        """
        if seq_ids is None:
            seq_ids = list(self.sequences.keys())

        # Validate sequence IDs
        missing_ids = [seq_id for seq_id in seq_ids if seq_id not in self.sequences]
        if missing_ids:
            raise ValueError(f"Sequence IDs not found: {missing_ids}")

        n_seqs = len(seq_ids)
        similarity_matrix = np.zeros((n_seqs, n_seqs))

        for i, seq_id1 in enumerate(seq_ids):
            for j, seq_id2 in enumerate(seq_ids):
                if i <= j:  # Only compute upper triangle
                    sim = self.jaccard_similarity(seq_id1, seq_id2)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Matrix is symmetric

        return similarity_matrix, seq_ids

    def save_sourmash_signatures(self, filepath: str):
        """
        Save all sourmash signatures to a file.

        Args:
            filepath: Output file path
        """
        import sourmash
        with open(filepath, 'w') as f:
            for signature in self.sourmash_signatures.values():
                sourmash.save_signatures([signature], f)

    def load_sourmash_signatures(self, filepath: str):
        """
        Load sourmash signatures from a file and add corresponding sequences.

        Args:
            filepath: Input file path with sourmash signatures

        Note:
            This will only load the signatures. Actual sequences need to be
            added separately using add_sequence().
        """
        import sourmash
        signatures = sourmash.load_file_as_signatures(filepath)

        for signature in signatures:
            seq_id = signature.name
            if seq_id:
                self.sourmash_signatures[seq_id] = signature

    def get_sequence_diversity_scores(self, reference_seq_ids: List[str]) -> Dict[str, float]:
        """
        Calculate diversity scores for all sequences compared to reference set.

        Args:
            reference_seq_ids: List of reference sequence IDs

        Returns:
            Dictionary mapping sequence IDs to diversity scores
        """
        diversity_scores = {}

        for seq_id in self.sequences.keys():
            similarities = []
            for ref_id in reference_seq_ids:
                if ref_id != seq_id and ref_id in self.sourmash_signatures:
                    sim = self.jaccard_similarity(seq_id, ref_id)
                    similarities.append(sim)

            # Diversity score is 1 - average similarity to reference set
            if similarities:
                diversity_scores[seq_id] = 1.0 - np.mean(similarities)
            else:
                diversity_scores[seq_id] = 1.0  # Maximum diversity if no references

        return diversity_scores
    
    def gc_content(self, seq_id):
        """calculate GC content (fraction 0-1)"""
        if seq_id not in self.sequences:
            raise KeyError(f"sequence {seq_id} not found")

        seq = self.sequences[seq_id]
        if not seq:
            return 0.0

        gc = seq.count('G') + seq.count('C')
        return gc / len(seq)
    
    def get_statistics(self):
        """get basic statistics about the profiler"""
        if not self.sequences:
            return {'error': 'no sequences loaded'}

        total_length = sum(len(seq) for seq in self.sequences.values())
        avg_length = total_length / len(self.sequences)

        # sourmash stats
        avg_minhash_size = 0
        if self.sourmash_signatures:
            total_hashes = sum(len(sig.minhash.hashes) for sig in self.sourmash_signatures.values())
            avg_minhash_size = total_hashes / len(self.sourmash_signatures)

        return {
            'n_sequences': len(self.sequences),
            'total_length': total_length,
            'avg_length': avg_length,
            'k': self.k,
            'scaled': self.scaled,
            'num_hashes': self.num_hashes,
            'avg_minhash_size': avg_minhash_size
        }
    
    def remove_sequence(self, seq_id):
        """remove a sequence from the profiler"""
        if seq_id not in self.sequences:
            raise KeyError(f"sequence {seq_id} not found")

        # clean up all the data structures
        del self.sequences[seq_id]
        if seq_id in self.sourmash_signatures:
            del self.sourmash_signatures[seq_id]
        if seq_id in self.sequence_metadata:
            del self.sequence_metadata[seq_id]
    
    def clear(self):
        """clear everything"""
        self.sequences.clear()
        self.sourmash_signatures.clear()
        self.sequence_metadata.clear()
    
    def copy(self):
        """make a copy of the profiler"""
        new_profiler = KmerProfiler(
            k=self.k,
            normalize=self.normalize,
            num_hashes=self.num_hashes,
            scaled=self.scaled
        )

        # copy all sequences
        for seq_id, sequence in self.sequences.items():
            metadata = self.sequence_metadata.get(seq_id)
            new_profiler.add_sequence(seq_id, sequence, metadata)

        return new_profiler
    
    def __len__(self):
        """Return number of sequences."""
        return len(self.sequences)
    
    def __contains__(self, seq_id: str):
        """Check if sequence ID exists."""
        return seq_id in self.sequences
    
    def __iter__(self):
        """Iterate over sequence IDs."""
        return iter(self.sequences.keys())
    
    def __repr__(self):
        return f"KmerProfiler(k={self.k}, {len(self.sequences)} seqs, scaled={self.scaled})"