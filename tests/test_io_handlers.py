"""
Unit tests for IO handlers
"""

import pytest
import tempfile
import os
from kadar.utils.io_handlers import load_fasta_sequences, save_fasta_sequences


class TestIOHandlers:
    """TD: change these across to be py.test fixtures"""

    def test_load_fasta_basic(self):
        """Test basic FASTA loading with real data"""
        # Test with the actual data file
        fasta_path = "data/GCA_000219625.1.fasta"
        if os.path.exists(fasta_path):
            sequences = load_fasta_sequences(fasta_path)

            assert len(sequences) > 0
            assert all(isinstance(seq_id, str) for seq_id in sequences.keys())
            assert all(isinstance(seq, str) for seq in sequences.values())

            # Check that sequences contain valid DNA characters
            for seq in sequences.values():
                valid_chars = set('ATGCRYSWKMBDHVN-')
                assert set(seq.upper()).issubset(valid_chars)

    def test_load_fasta_small_file(self):
        """Test FASTA loading with small test file"""
        fasta_path = "data/GCF_004022145.1_Paevar1_genomic.fasta"
        if os.path.exists(fasta_path):
            sequences = load_fasta_sequences(fasta_path)

            assert len(sequences) > 0
            # Check first sequence
            first_seq = list(sequences.values())[0]
            assert len(first_seq) > 1000  # Should be substantial

    def test_save_and_load_fasta(self):
        """Test saving and loading FASTA files"""
        test_sequences = {
            "seq1": "ATCGATCGATCGATCG",
            "seq2": "GCTAGCTAGCTAGCTA",
            "seq3": "AAAAAATTTTTTGGGGGGCCCCCC"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            temp_path = f.name

        try:
            # Save sequences
            save_fasta_sequences(test_sequences, temp_path)
            assert os.path.exists(temp_path)

            # Load them back
            loaded_sequences = load_fasta_sequences(temp_path)

            assert len(loaded_sequences) == len(test_sequences)
            assert set(loaded_sequences.keys()) == set(test_sequences.keys())

            # Check that sequences match (should be uppercase)
            for seq_id, original_seq in test_sequences.items():
                assert loaded_sequences[seq_id] == original_seq.upper()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_fasta_with_line_wrapping(self):
        """Test FASTA saving with line length control"""
        long_sequence = "A" * 200  # 200 bp sequence
        test_sequences = {"long_seq": long_sequence}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            temp_path = f.name

        try:
            # Save with 50 character lines
            save_fasta_sequences(test_sequences, temp_path, line_length=50)

            # Read the file and check line lengths
            with open(temp_path, 'r') as f:
                lines = f.readlines()

            # First line should be header
            assert lines[0].startswith('>')

            # Sequence lines should be ≤ 50 characters (plus newline)
            for line in lines[1:]:
                assert len(line.strip()) <= 50

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent files"""
        with pytest.raises(FileNotFoundError):
            load_fasta_sequences("nonexistent_file.fasta")

    def test_empty_fasta_file(self):
        """Test handling of empty FASTA file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            temp_path = f.name
            # Write nothing

        try:
            with pytest.raises(ValueError, match="No sequences found"):
                load_fasta_sequences(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_malformed_fasta(self):
        """Test handling of malformed FASTA file"""
        malformed_content = """
        This is not a FASTA file
        It has no headers
        ATCGATCG
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(malformed_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                load_fasta_sequences(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_fasta_with_empty_sequences(self):
        """Test FASTA file with empty sequences"""
        fasta_content = """>seq1

>seq2
ATCGATCG
>seq3

"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_path = f.name

        try:
            sequences = load_fasta_sequences(temp_path)

            # Should load seq2, others might be empty
            assert "seq2" in sequences
            assert sequences["seq2"] == "ATCGATCG"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_fasta_with_special_characters(self):
        """Test FASTA with special DNA characters"""
        fasta_content = """>seq_with_special
ATCGRYSWKMBDHVN-ATCG
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_path = f.name

        try:
            sequences = load_fasta_sequences(temp_path)
            assert "seq_with_special" in sequences
            # Should preserve special characters
            assert "R" in sequences["seq_with_special"]
            assert "N" in sequences["seq_with_special"]
            assert "-" in sequences["seq_with_special"]

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_multi_line_sequences(self):
        """Test FASTA with multi-line sequences"""
        fasta_content = """>multi_line_seq
ATCGATCGATCGATCG
GCTAGCTAGCTAGCTA
AAAAAATTTTTTGGGG
CCCCCCTTTTTTAAAA
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_path = f.name

        try:
            sequences = load_fasta_sequences(temp_path)
            expected = "ATCGATCGATCGATCGGCTAGCTAGCTAGCTAAAAAAATTTTTTGGGGCCCCCCTTTTTTAAAA"
            assert sequences["multi_line_seq"] == expected

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)