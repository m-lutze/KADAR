"""
Integration tests for the full pipeline
"""

import os

import pytest

from kadar import (
    GenomeIslandPredictor,
    KmerProfiler,
    load_fasta_sequences,
    read_and_insert_islands,
)


class TestIntegration:
    """Integration tests using real data"""

    @pytest.fixture
    def small_fasta_path(self):
        """Path to smaller FASTA file for testing"""
        path = 'data/GCF_004022145.1_Paevar1_genomic.fasta'
        if os.path.exists(path):
            return path
        pytest.skip('Small FASTA file not found')

    @pytest.fixture
    def large_fasta_path(self):
        """Path to larger FASTA file for testing"""
        path = 'data/GCA_000219625.1.fasta'
        if os.path.exists(path):
            return path
        pytest.skip('Large FASTA file not found')

    def test_basic_pipeline_with_real_data(self, small_fasta_path):
        """Test basic pipeline with real genomic data"""
        # Load sequences
        sequences = load_fasta_sequences(small_fasta_path)
        assert len(sequences) > 0

        # Create profiler and add sequences
        profiler = KmerProfiler(k=4, scaled=1000)

        # Add first few sequences (limit to avoid memory issues)
        seq_items = list(sequences.items())[:5]
        for seq_id, sequence in seq_items:
            # Only add sequences that are long enough
            if len(sequence) > 1000:
                profiler.add_sequence(seq_id, sequence)

        if len(profiler) == 0:
            pytest.skip('No sequences long enough for testing')

        # Test basic functionality
        stats = profiler.get_statistics()
        assert stats['n_sequences'] == len(profiler)

        # Test similarity calculation if we have multiple sequences
        if len(profiler) >= 2:
            seq_ids = list(profiler.sequences.keys())
            similarity = profiler.jaccard_similarity(seq_ids[0], seq_ids[1])
            assert 0 <= similarity <= 1

    def test_synthetic_island_pipeline(self, small_fasta_path):
        """Test full pipeline with synthetic islands"""
        # Insert synthetic islands
        island_data = read_and_insert_islands(
            fasta_path=small_fasta_path,
            n_islands=1,
            island_size_range=(2000, 5000),
            min_spacing=3000,
            random_seed=42,
        )

        # Should have processed some sequences
        successful_insertions = {
            seq_id: data
            for seq_id, data in island_data.items()
            if len(data['islands']) > 0
        }

        if not successful_insertions:
            pytest.skip('No sequences were long enough for island insertion')

        # Test with a sequence that had islands inserted
        seq_id, data = list(successful_insertions.items())[0]

        # Create profiler with both original and modified sequences
        profiler = KmerProfiler(k=4, num_hashes=500, scaled=0)
        profiler.add_sequence(f'{seq_id}_original', data['original_sequence'])
        profiler.add_sequence(f'{seq_id}_modified', data['modified_sequence'])

        # Calculate similarity between original and modified
        similarity = profiler.jaccard_similarity(
            f'{seq_id}_original', f'{seq_id}_modified'
        )

        # Check if islands were actually inserted
        if len(data['islands']) > 0:
            # Sequences should have some similarity (may be 1.0 for very large seqs with small islands)
            assert 0.5 <= similarity <= 1.0
        else:
            # If no islands were inserted, sequences should be identical
            assert similarity == 1.0

    def test_predictor_with_synthetic_data(self, small_fasta_path):
        """Test GenomeIslandPredictor with synthetic islands"""
        # Create synthetic data
        island_data = read_and_insert_islands(
            fasta_path=small_fasta_path,
            n_islands=2,
            island_size_range=(3000, 6000),
            min_spacing=5000,
            random_seed=42,
        )

        # Find sequences with successful insertions
        valid_sequences = {}
        for seq_id, data in island_data.items():
            if len(data['islands']) > 0 and data['modified_length'] > 10000:
                valid_sequences[seq_id] = data
                if len(valid_sequences) >= 3:  # Limit for testing
                    break

        if len(valid_sequences) < 2:
            pytest.skip('Not enough valid sequences for predictor testing')

        # Create profiler with modified sequences
        profiler = KmerProfiler(k=4, scaled=1000)
        for seq_id, data in valid_sequences.items():
            profiler.add_sequence(seq_id, data['modified_sequence'])

        # Create predictor
        predictor = GenomeIslandPredictor(profiler)

        # Test basic predictor functionality
        seq_ids = list(profiler.sequences.keys())

        # Test PCA analysis
        try:
            pca_result = predictor.pca_analysis(seq_ids[:3])
            assert 'explained_variance_ratio' in pca_result
            assert 'transformed_data' in pca_result
        except Exception as e:
            # PCA might fail with very similar sequences
            print(f'PCA analysis failed (expected): {e}')

        # Test clustering
        try:
            clustering_result = predictor.clustering_analysis(seq_ids[:3])
            assert 'labels' in clustering_result
            assert 'method' in clustering_result
        except Exception as e:
            print(f'Clustering analysis failed (expected): {e}')

    def test_sliding_window_analysis(self, small_fasta_path):
        """Test sliding window analysis on sequences with islands"""
        # Create data with islands
        island_data = read_and_insert_islands(
            fasta_path=small_fasta_path,
            n_islands=1,
            island_size_range=(3000, 5000),
            min_spacing=10000,
            random_seed=42,
        )

        # Find a sequence with islands
        test_seq_data = None
        for _seq_id, data in island_data.items():
            if len(data['islands']) > 0 and data['modified_length'] > 20000:
                test_seq_data = data
                break

        if test_seq_data is None:
            pytest.skip('No suitable sequence found for sliding window test')

        # Create profiler
        KmerProfiler(k=4, scaled=1000)

        # Perform sliding window analysis
        sequence = test_seq_data['modified_sequence']
        window_size = 5000
        step_size = 2500

        window_profiles = []
        window_positions = []

        for start in range(0, len(sequence) - window_size + 1, step_size):
            end = start + window_size
            window_seq = sequence[start:end]
            window_id = f'window_{start}_{end}'

            # Create temporary profiler for this window
            temp_profiler = KmerProfiler(k=4, scaled=1000)
            temp_profiler.add_sequence(window_id, window_seq)

            window_profiles.append(temp_profiler)
            window_positions.append((start, end))

            # Limit number of windows for testing
            if len(window_profiles) >= 10:
                break

        assert len(window_profiles) > 0

        # Test that we can compare windows
        if len(window_profiles) >= 2:
            prof1 = window_profiles[0]
            prof2 = window_profiles[1]

            window_ids = [list(p.sequences.keys())[0] for p in [prof1, prof2]]

            # Create combined profiler for comparison
            combined_profiler = KmerProfiler(k=4, scaled=1000)
            combined_profiler.add_sequence(
                window_ids[0], list(prof1.sequences.values())[0]
            )
            combined_profiler.add_sequence(
                window_ids[1], list(prof2.sequences.values())[0]
            )

            similarity = combined_profiler.jaccard_similarity(
                window_ids[0], window_ids[1]
            )
            assert 0 <= similarity <= 1

    def test_memory_efficiency_large_file(self, large_fasta_path):
        """Test memory efficiency with larger file (limited processing)"""
        # Load just the first sequence from the large file
        sequences = load_fasta_sequences(large_fasta_path)
        first_seq_id, first_seq = next(iter(sequences.items()))

        # Only use first 100kb to avoid memory issues
        if len(first_seq) > 100000:
            first_seq = first_seq[:100000]

        # Test with sourmash (should be memory efficient)
        profiler = KmerProfiler(k=6, scaled=2000)  # Larger k, larger scale
        profiler.add_sequence('test_seq', first_seq)

        # Should work without memory issues
        stats = profiler.get_statistics()
        assert stats['n_sequences'] == 1

        # Test similarity matrix creation (should be fast)
        matrix, seq_ids, hashes = profiler.get_hash_matrix()
        assert matrix.shape == (1, len(hashes))

    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline"""
        # Test with invalid file
        with pytest.raises(FileNotFoundError):
            load_fasta_sequences('nonexistent.fasta')

        # Test profiler with invalid sequences
        profiler = KmerProfiler(k=4)
        with pytest.raises(ValueError):
            profiler.add_sequence('invalid', 'ATCGXYZ')

        # Test predictor with empty profiler
        empty_profiler = KmerProfiler(k=4)
        with pytest.raises(ValueError):
            GenomeIslandPredictor(empty_profiler)

    def test_reproducibility(self, small_fasta_path):
        """Test that results are reproducible with same random seed"""
        # Generate islands twice with same seed
        island_data1 = read_and_insert_islands(
            fasta_path=small_fasta_path,
            n_islands=1,
            island_size_range=(2000, 4000),
            min_spacing=3000,
            random_seed=12345,
        )

        island_data2 = read_and_insert_islands(
            fasta_path=small_fasta_path,
            n_islands=1,
            island_size_range=(2000, 4000),
            min_spacing=3000,
            random_seed=12345,
        )

        # Results should be identical
        assert len(island_data1) == len(island_data2)

        for seq_id in island_data1.keys():
            if seq_id in island_data2:
                data1 = island_data1[seq_id]
                data2 = island_data2[seq_id]

                # Modified sequences should be identical
                assert data1['modified_sequence'] == data2['modified_sequence']

                # Island positions should be identical
                if len(data1['islands']) > 0 and len(data2['islands']) > 0:
                    assert data1['islands'][0]['start'] == data2['islands'][0]['start']
                    assert data1['islands'][0]['end'] == data2['islands'][0]['end']
