"""
Functions for loading and saving genome sequence data
"""

import os
from typing import Dict, List, Optional, Union
import gzip
import json
import pickle

def load_fasta_sequences(filepath: str) -> Dict[str, str]:
    """
    Load sequences from a FASTA file.
    
    Args:
        filepath: Path to FASTA file (can be .gz compressed)
        
    Returns:
        Dictionary of sequence_id -> sequence
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    sequences = {}
    current_id = None
    current_seq = []
    
    # Determine if file is compressed
    open_func = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'
    
    try:
        with open_func(filepath, mode) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        if not current_seq:
                            print(f"Warning: Empty sequence for {current_id}")
                        sequences[current_id] = ''.join(current_seq)
                    
                    # Start new sequence
                    current_id = line[1:]  # Remove '>' character
                    current_seq = []
                    
                elif current_id is not None:
                    # Validate sequence characters
                    valid_chars = set('ATGCRYSWKMBDHVN-')
                    invalid_chars = set(line.upper()) - valid_chars
                    if invalid_chars:
                        print(f"Warning: Invalid characters {invalid_chars} in line {line_num}")
                    
                    current_seq.append(line.upper())
                else:
                    raise ValueError(f"Sequence data found before header at line {line_num}")
            
            # Don't forget the last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
    
    except Exception as e:
        raise ValueError(f"Error reading FASTA file: {e}")
    
    if not sequences:
        raise ValueError("No sequences found in FASTA file")
    
    return sequences


def save_fasta_sequences(sequences: Dict[str, str], filepath: str, 
                        line_length: int = 80, compress: bool = False):
    """
    Save sequences to a FASTA file.
    
    Args:
        sequences: Dictionary of sequence_id -> sequence
        filepath: Output file path
        line_length: Maximum line length for sequence data
        compress: Whether to compress output with gzip
    """
    if compress and not filepath.endswith('.gz'):
        filepath += '.gz'
    
    open_func = gzip.open if compress else open
    mode = 'wt' if compress else 'w'
    
    with open_func(filepath, mode) as f:
        for seq_id, sequence in sequences.items():
            f.write(f'>{seq_id}\n')
            
            # Write sequence in chunks of line_length
            for i in range(0, len(sequence), line_length):
                f.write(sequence[i:i + line_length] + '\n')

__all__ = ['KmerProfiler', 'GenomeIslandPredictor']
