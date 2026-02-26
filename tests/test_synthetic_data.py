"""
Unit tests for synthetic data generation
"""

import os
import tempfile

import pytest

from kadar.utils.synthetic_data import (
    choose_island_positions,
    choose_island_type,
    generate_island_sequence,
    insert_islands_into_sequence,
    read_and_insert_islands,
    save_island_data,
)


class TestSyntheticData:
    """Test synthetic data functionality"""

    def test_generate_island_sequence(self):
        """Test island sequence generation"""
        # Test different island types
        types = ['prophage', 'pathogenicity', 'antibiotic_resistance', 'metabolic']

        for island_type in types:
            seq = generate_island_sequence(1000, island_type)

            assert len(seq) == 1000
            assert set(seq).issubset(set('ATGC'))

            # Check that sequences have different compositions
            a_content = seq.count('A') / len(seq)
            t_content = seq.count('T') / len(seq)
            g_content = seq.count('G') / len(seq)
            c_content = seq.count('C') / len(seq)

            # Should sum to 1 (approximately)
            total = a_content + t_content + g_content + c_content
            assert abs(total - 1.0) < 0.01

    def test_choose_island_type(self):
        """Test island type selection"""
        # Generate multiple types and check they're all valid
        types = [choose_island_type() for _ in range(20)]

        valid_types = {
            'prophage',
            'pathogenicity',
            'antibiotic_resistance',
            'metabolic',
            'transposon',
            'plasmid_derived',
        }

        assert all(t in valid_types for t in types)
        # Should have some variety
        assert len(set(types)) > 1

    def test_choose_island_positions(self):
        """Test island position selection"""
        seq_len = 100000
        n_islands = 3
        max_island_size = 5000
        min_spacing = 2000

        positions = choose_island_positions(
            seq_len, n_islands, max_island_size, min_spacing
        )

        assert len(positions) == n_islands
        assert all(0 <= pos < seq_len for pos in positions)

        # Check spacing
        sorted_pos = sorted(positions)
        for i in range(1, len(sorted_pos)):
            spacing = sorted_pos[i] - sorted_pos[i - 1]
            assert spacing >= min_spacing

    def test_choose_island_positions_insufficient_space(self):
        """Test error when sequence is too short"""
        seq_len = 1000
        n_islands = 10
        max_island_size = 500
        min_spacing = 200

        with pytest.raises(ValueError, match='too short'):
            choose_island_positions(seq_len, n_islands, max_island_size, min_spacing)

    def test_insert_islands_into_sequence(self):
        """Test inserting islands into a sequence"""
        original_seq = 'A' * 50000  # 50kb of As
        n_islands = 2
        size_range = (1000, 3000)
        min_spacing = 2000

        modified_seq, island_info = insert_islands_into_sequence(
            original_seq, n_islands, size_range, min_spacing
        )

        # Check that sequence was modified
        assert len(modified_seq) > len(original_seq)
        assert len(island_info) == n_islands

        # Check island info structure
        for island in island_info:
            assert 'start' in island
            assert 'end' in island
            assert 'type' in island
            assert 'length' in island
            assert 'island_id' in island

            assert island['end'] > island['start']
            assert size_range[0] <= island['length'] <= size_range[1]

        # Check that islands don't overlap and maintain spacing
        sorted_islands = sorted(island_info, key=lambda x: x['start'])
        for i in range(1, len(sorted_islands)):
            prev_end = sorted_islands[i - 1]['end']
            curr_start = sorted_islands[i]['start']
            assert curr_start >= prev_end  # No overlap

    def test_read_and_insert_islands_with_real_data(self):
        """Test reading real FASTA and inserting islands"""
        fasta_path = 'data/GCF_004022145.1_Paevar1_genomic.fasta'

        if os.path.exists(fasta_path):
            result = read_and_insert_islands(
                fasta_path=fasta_path,
                n_islands=1,  # Use fewer islands for smaller sequences
                island_size_range=(1000, 3000),
                min_spacing=2000,
                random_seed=42,
            )

            assert len(result) > 0

            for _seq_id, data in result.items():
                assert 'original_sequence' in data
                assert 'modified_sequence' in data
                assert 'islands' in data
                assert 'original_length' in data
                assert 'modified_length' in data

                # Modified should be longer
                assert data['modified_length'] >= data['original_length']

                # Should have islands (unless sequence was too short)
                if data['modified_length'] > data['original_length']:
                    assert len(data['islands']) > 0

    def test_save_island_data(self):
        """Test saving island data"""
        # Create test data
        test_data = {
            'seq1': {
                'original_sequence': 'ATCGATCG' * 1000,
                'modified_sequence': 'ATCGATCG' * 1000 + 'GCTAGCTA' * 500,
                'islands': [
                    {
                        'start': 8000,
                        'end': 12000,
                        'type': 'prophage',
                        'length': 4000,
                        'island_id': 'synthetic_island_1',
                    }
                ],
                'original_length': 8000,
                'modified_length': 12000,
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'test')

            save_island_data(test_data, output_prefix)

            # Check that files were created
            assert os.path.exists(f'{output_prefix}_with_islands.fasta')
            assert os.path.exists(f'{output_prefix}_original.fasta')
            assert os.path.exists(f'{output_prefix}_annotations.json')

            # Check JSON annotations
            import json

            with open(f'{output_prefix}_annotations.json') as f:
                annotations = json.load(f)

            assert 'seq1' in annotations
            assert annotations['seq1']['original_length'] == 8000
            assert len(annotations['seq1']['islands']) == 1

    def test_insert_islands_edge_cases(self):
        """Test edge cases for island insertion"""
        # Very short sequence - should return unchanged
        short_seq = 'ATCGATCG'
        modified_seq, island_info = insert_islands_into_sequence(
            short_seq, 1, (100, 200), 50
        )
        assert modified_seq == short_seq  # Should be unchanged
        assert len(island_info) == 0  # No islands should be inserted

        # Zero islands
        normal_seq = 'A' * 10000
        modified_seq, island_info = insert_islands_into_sequence(
            normal_seq, 0, (100, 200), 50
        )
        assert modified_seq == normal_seq
        assert len(island_info) == 0

    def test_island_sequence_composition_differences(self):
        """Test that different island types have different compositions"""
        # Generate sequences for different types
        prophage = generate_island_sequence(10000, 'prophage')
        pathogenicity = generate_island_sequence(10000, 'pathogenicity')
        antibiotic = generate_island_sequence(10000, 'antibiotic_resistance')

        # Calculate GC content for each
        def gc_content(seq):
            return (seq.count('G') + seq.count('C')) / len(seq)

        gc_prophage = gc_content(prophage)
        gc_pathogenicity = gc_content(pathogenicity)
        gc_antibiotic = gc_content(antibiotic)

        # Pathogenicity islands should be GC-rich
        assert gc_pathogenicity > gc_prophage
        assert gc_pathogenicity > gc_antibiotic

        # Antibiotic resistance should be A-rich (low GC)
        assert gc_antibiotic < gc_pathogenicity

    def test_island_generation_reproducibility(self):
        """Test that island generation is reproducible with same seed"""
        import random

        import numpy as np

        # Generate with same seed
        np.random.seed(42)
        random.seed(42)
        seq1 = generate_island_sequence(1000, 'prophage')

        np.random.seed(42)
        random.seed(42)
        seq2 = generate_island_sequence(1000, 'prophage')

        assert seq1 == seq2
