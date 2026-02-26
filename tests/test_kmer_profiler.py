"""
Unit tests for KmerProfiler class
"""

import numpy as np
import pytest

from kadar.core.kmer_profiler import KmerProfiler


class TestKmerProfiler:
    """TD: change these across to be py.test fixtures"""

    def test_init_basic(self):
        """Test basic initialization"""
        profiler = KmerProfiler(k=4, scaled=1000)
        assert profiler.k == 4
        assert profiler.scaled == 1000
        assert len(profiler) == 0

    def test_init_parameters(self):
        """Test initialization with different parameters"""
        profiler = KmerProfiler(k=6, normalize=False, num_hashes=500, scaled=2000)
        assert profiler.k == 6
        assert not profiler.normalize
        assert profiler.num_hashes == 500
        assert profiler.scaled == 2000

    def test_invalid_k(self):
        """Test that invalid k values raise errors"""
        with pytest.raises(ValueError):
            KmerProfiler(k=0)

        with pytest.raises(ValueError):
            KmerProfiler(k=-1)

    def test_add_sequence_basic(self):
        """Test adding sequences"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')

        assert len(profiler) == 1
        assert 'seq1' in profiler
        assert 'seq1' in profiler.sequences
        assert 'seq1' in profiler.sourmash_signatures

    def test_add_sequence_with_metadata(self):
        """Test adding sequences with metadata"""
        profiler = KmerProfiler(k=4)
        metadata = {'type': 'host', 'length': 16}
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG', metadata)

        assert profiler.sequence_metadata['seq1'] == metadata

    def test_add_duplicate_sequence(self):
        """Test that duplicate sequence IDs raise errors"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')

        with pytest.raises(ValueError, match='already exists'):
            profiler.add_sequence('seq1', 'GCTAGCTAGCTAGCTA')

    def test_clean_sequence_validation(self):
        """Test sequence validation and cleaning"""
        profiler = KmerProfiler(k=4)

        # Valid sequence should work
        profiler.add_sequence('valid', 'atcgatcg')  # lowercase should be converted
        assert profiler.sequences['valid'] == 'ATCGATCG'

        # Invalid characters should raise error
        with pytest.raises(ValueError, match='invalid characters'):
            profiler.add_sequence('invalid', 'ATCGXYZ')

        # Empty sequence should raise error
        with pytest.raises(ValueError, match='empty sequence'):
            profiler.add_sequence('empty', '')

    def test_sequence_with_ns(self):
        """Test handling sequences with N characters"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('with_n', 'ATCGNATCG')
        assert 'with_n' in profiler.sequences

    def test_sourmash_signature_creation(self):
        """Test that sourmash signatures are created"""
        profiler = KmerProfiler(k=4, scaled=1000)
        profiler.add_sequence('test', 'ATCGATCGATCGATCGATCGATCGATCG')

        sig = profiler.get_signature('test')
        assert sig is not None
        assert sig.name == 'test'

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation"""
        # Use num_hashes instead of scaled for more predictable results
        profiler = KmerProfiler(k=4, num_hashes=100, scaled=0)

        # Use longer sequences to ensure adequate k-mer sampling
        seq1 = 'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG' * 10
        seq2 = (
            'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG' * 10
        )  # identical
        seq3 = (
            'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' * 10
        )  # different

        profiler.add_sequence('seq1', seq1)
        profiler.add_sequence('seq2', seq2)
        profiler.add_sequence('seq3', seq3)

        # Identical sequences should have similarity close to 1
        sim_identical = profiler.jaccard_similarity('seq1', 'seq2')
        assert sim_identical >= 0.9  # Should be very high

        # Different sequences should have lower similarity
        sim_different = profiler.jaccard_similarity('seq1', 'seq3')
        assert sim_different < sim_identical

    def test_containment(self):
        """Test containment calculation"""
        profiler = KmerProfiler(k=4, scaled=1000)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTAGCTAGCTAGCTA')

        containment = profiler.containment('seq1', 'seq2')
        assert 0 <= containment <= 1

    def test_hash_matrix(self):
        """Test hash matrix generation"""
        profiler = KmerProfiler(k=4, scaled=1000)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTAGCTAGCTAGCTA')

        matrix, seq_ids, hashes = profiler.get_hash_matrix()

        assert matrix.shape[0] == 2  # Two sequences
        assert len(seq_ids) == 2
        assert seq_ids == ['seq1', 'seq2']
        assert matrix.dtype == np.float64

        # Matrix should contain only 0s and 1s (presence/absence)
        assert np.all((matrix == 0) | (matrix == 1))

    def test_hash_matrix_subset(self):
        """Test hash matrix with sequence subset"""
        profiler = KmerProfiler(k=4, scaled=1000)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTAGCTAGCTAGCTA')
        profiler.add_sequence('seq3', 'AAAAAAAAAAAAAAAAAAAAAAAAAAAA')

        matrix, seq_ids, hashes = profiler.get_hash_matrix(['seq1', 'seq3'])

        assert matrix.shape[0] == 2
        assert seq_ids == ['seq1', 'seq3']

    def test_gc_content(self):
        """Test GC content calculation"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('at_rich', 'AAAAAATTTTTTAAAAAATTTTT')
        profiler.add_sequence('gc_rich', 'GGGGGCCCCCGGGGGGCCCCCC')
        profiler.add_sequence('balanced', 'ATCGATCGATCGATCGATCG')

        gc_at = profiler.gc_content('at_rich')
        gc_gc = profiler.gc_content('gc_rich')
        gc_bal = profiler.gc_content('balanced')

        assert gc_at < gc_bal < gc_gc
        assert 0 <= gc_at <= 1
        assert 0 <= gc_gc <= 1

    def test_get_statistics(self):
        """Test statistics generation"""
        profiler = KmerProfiler(k=4, scaled=1000)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTAGCTAGCTAGCTA')

        stats = profiler.get_statistics()

        assert stats['n_sequences'] == 2
        assert stats['k'] == 4
        assert stats['scaled'] == 1000
        assert 'total_length' in stats
        assert 'avg_length' in stats

    def test_remove_sequence(self):
        """Test sequence removal"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTA')

        assert len(profiler) == 2

        profiler.remove_sequence('seq1')

        assert len(profiler) == 1
        assert 'seq1' not in profiler
        assert 'seq2' in profiler

        # Should raise error for non-existent sequence
        with pytest.raises(KeyError):
            profiler.remove_sequence('nonexistent')

    def test_clear(self):
        """Test clearing all sequences"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTA')

        assert len(profiler) == 2

        profiler.clear()

        assert len(profiler) == 0
        assert len(profiler.sequences) == 0
        assert len(profiler.sourmash_signatures) == 0

    def test_copy(self):
        """Test copying profiler"""
        profiler = KmerProfiler(k=4, scaled=1000)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG', {'type': 'test'})

        profiler_copy = profiler.copy()

        assert profiler_copy.k == profiler.k
        assert profiler_copy.scaled == profiler.scaled
        assert len(profiler_copy) == len(profiler)
        assert 'seq1' in profiler_copy

        # Should be independent copies
        profiler_copy.add_sequence('seq2', 'GCTAGCTAGCTAGCTA')
        assert 'seq2' not in profiler
        assert 'seq2' in profiler_copy

    def test_iter_and_contains(self):
        """Test iteration and containment checks"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')
        profiler.add_sequence('seq2', 'GCTAGCTAGCTAGCTA')

        # Test iteration
        seq_ids = list(profiler)
        assert set(seq_ids) == {'seq1', 'seq2'}

        # Test containment
        assert 'seq1' in profiler
        assert 'seq2' in profiler
        assert 'seq3' not in profiler

    def test_repr(self):
        """Test string representation"""
        profiler = KmerProfiler(k=6, scaled=2000)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')

        repr_str = repr(profiler)
        assert 'k=6' in repr_str
        assert '1 seqs' in repr_str
        assert 'scaled=2000' in repr_str

    def test_error_handling_empty_profiler(self):
        """Test error handling with empty profiler"""
        profiler = KmerProfiler(k=4)

        # Should handle empty profiler gracefully
        with pytest.raises(ValueError, match='no sequences'):
            profiler.get_hash_matrix()

        stats = profiler.get_statistics()
        assert 'error' in stats

    def test_missing_sequence_errors(self):
        """Test errors for missing sequences"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')

        with pytest.raises(KeyError):
            profiler.get_signature('nonexistent')

        with pytest.raises(KeyError):
            profiler.gc_content('nonexistent')

    def test_get_hash_matrix_missing_sequences(self):
        """Test hash matrix with missing sequence IDs"""
        profiler = KmerProfiler(k=4)
        profiler.add_sequence('seq1', 'ATCGATCGATCGATCG')

        with pytest.raises(ValueError, match='not found'):
            profiler.get_hash_matrix(['seq1', 'nonexistent'])
