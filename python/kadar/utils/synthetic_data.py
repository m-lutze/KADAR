"""
Functions to read existing genomic data and insert synthetic genome islands
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io_handlers import load_fasta_sequences


def read_and_insert_islands(
    fasta_path: str,
    n_islands: int = 3,
    island_size_range: Tuple[int, int] = (5000, 20000),
    min_spacing: int = 10000,
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """
    Read existing genome and insert synthetic islands

    Args:
        fasta_path: path to existing genome fasta file
        n_islands: number of islands to insert
        island_size_range: (min, max) size of islands in bp
        min_spacing: minimum spacing between islands
        random_seed: for reproducible results

    Returns:
        dict with modified sequences and island locations
    """
    if random_seed:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # load existing sequences
    print(f'Reading sequences from {fasta_path}')
    original_seqs = load_fasta_sequences(fasta_path)

    results = {}

    for seq_id, sequence in original_seqs.items():
        print(f'Processing {seq_id} (length: {len(sequence)})')

        # insert islands into this sequence
        modified_seq, island_info = insert_islands_into_sequence(
            sequence, n_islands, island_size_range, min_spacing
        )

        results[seq_id] = {
            'original_sequence': sequence,
            'modified_sequence': modified_seq,
            'islands': island_info,
            'original_length': len(sequence),
            'modified_length': len(modified_seq),
        }

    return results


def insert_islands_into_sequence(
    sequence: str, n_islands: int, size_range: Tuple[int, int], min_spacing: int = 10000
) -> Tuple[str, List[Dict]]:
    """
    Insert synthetic islands into a single sequence

    Args:
        sequence: original DNA sequence
        n_islands: number of islands to insert
        size_range: (min, max) island sizes
        min_spacing: minimum spacing between insertions

    Returns:
        (modified_sequence, island_info_list)
    """
    seq_len = len(sequence)

    # Check if sequence is long enough for requested islands
    min_required = n_islands * (size_range[1] + min_spacing)
    if seq_len < min_required:
        # Return sequence unchanged if too short
        return sequence, []

    # figure out where to put islands
    try:
        positions = choose_island_positions(
            seq_len, n_islands, size_range[1], min_spacing
        )
    except ValueError:
        # If position choosing fails, return unchanged sequence
        return sequence, []

    # generate islands
    islands = []
    for i, pos in enumerate(positions):
        island_size = random.randint(size_range[0], size_range[1])
        island_type = choose_island_type()
        island_seq = generate_island_sequence(island_size, island_type)

        islands.append(
            {
                'position': pos,
                'size': island_size,
                'type': island_type,
                'sequence': island_seq,
                'island_id': f'synthetic_island_{i + 1}',
            }
        )

    # insert islands (go backwards so positions don't shift)
    modified_seq = sequence
    island_info = []

    for island in sorted(islands, key=lambda x: x['position'], reverse=True):
        pos = island['position']
        island_seq = island['sequence']

        # insert the island
        modified_seq = modified_seq[:pos] + island_seq + modified_seq[pos:]

        # record where it ended up
        island_info.append(
            {
                'start': pos,
                'end': pos + len(island_seq),
                'type': island['type'],
                'length': len(island_seq),
                'island_id': island['island_id'],
            }
        )

    # reverse the list since we went backwards
    island_info.reverse()

    return modified_seq, island_info


def choose_island_positions(
    seq_len: int, n_islands: int, max_island_size: int, min_spacing: int
) -> List[int]:
    """
    Choose random positions for island insertion

    Make sure there's enough space and proper spacing
    """
    # need to leave room for islands and spacing
    usable_length = seq_len - (n_islands * max_island_size) - (n_islands * min_spacing)

    if usable_length < 0:
        raise ValueError(f'Sequence too short for {n_islands} islands with spacing')

    # generate random positions with spacing
    positions = []

    for i in range(n_islands):
        if i == 0:
            # first island can go anywhere in first section
            max_pos = usable_length // n_islands
            if max_pos < min_spacing:
                max_pos = min_spacing + 100  # ensure valid range
            max_pos = min(max_pos, seq_len - max_island_size)
            if max_pos <= min_spacing:
                continue  # Skip this island if no valid position
            pos = random.randint(min_spacing, max_pos)
        else:
            # subsequent islands need spacing from previous
            last_pos = positions[-1]
            min_pos = last_pos + max_island_size + min_spacing
            max_pos = min_pos + (usable_length // n_islands)
            max_pos = min(max_pos, seq_len - max_island_size)
            if max_pos <= min_pos:
                continue  # Skip this island if no valid position
            pos = random.randint(min_pos, max_pos)

        positions.append(pos)

    return sorted(positions)


def choose_island_type() -> str:
    """
    Randomly choose an island type
    """
    island_types = [
        'prophage',
        'pathogenicity',
        'antibiotic_resistance',
        'metabolic',
        'transposon',
        'plasmid_derived',
    ]
    return random.choice(island_types)


def generate_island_sequence(length: int, island_type: str) -> str:
    """
    Generate synthetic island sequence with appropriate composition

    Different island types have different nucleotide biases
    """
    # define composition patterns for different island types
    compositions = {
        'prophage': {'A': 0.32, 'T': 0.33, 'G': 0.17, 'C': 0.18},  # slightly AT rich
        'pathogenicity': {'A': 0.15, 'T': 0.15, 'G': 0.35, 'C': 0.35},  # GC rich
        'antibiotic_resistance': {'A': 0.40, 'T': 0.20, 'G': 0.20, 'C': 0.20},  # A rich
        'metabolic': {'A': 0.28, 'T': 0.27, 'G': 0.23, 'C': 0.22},  # balanced
        'transposon': {'A': 0.35, 'T': 0.35, 'G': 0.15, 'C': 0.15},  # AT rich
        'plasmid_derived': {'A': 0.30, 'T': 0.25, 'G': 0.25, 'C': 0.20},  # mixed
        'default': {'A': 0.35, 'T': 0.35, 'G': 0.15, 'C': 0.15},
    }

    comp = compositions.get(island_type, compositions['default'])

    # add some random variation to make it more realistic
    variation = np.random.normal(0, 0.02, 4)
    probs = np.array(list(comp.values())) + variation
    probs = np.abs(probs)  # no negative probs
    probs = probs / np.sum(probs)  # normalize

    # generate sequence
    bases = list(comp.keys())
    island_seq = ''.join(np.random.choice(bases, length, p=probs))

    return island_seq


def add_island_features(
    sequence: str, island_type: str, add_repeats: bool = True, add_genes: bool = True
) -> str:
    """
    Add realistic features to island sequences

    Args:
        sequence: base island sequence
        island_type: type of island
        add_repeats: whether to add repeat elements
        add_genes: whether to add gene-like patterns
    """
    modified_seq = sequence

    # add direct repeats at ends (common in real islands)
    if add_repeats and len(sequence) > 100:
        repeat_len = random.randint(10, 50)
        repeat_seq = generate_random_sequence(repeat_len)
        modified_seq = repeat_seq + modified_seq + repeat_seq

    # add some gene-like patterns
    if add_genes and len(modified_seq) > 500:
        modified_seq = insert_gene_patterns(modified_seq, island_type)

    return modified_seq


def generate_random_sequence(length: int, composition: Dict[str, float] = None) -> str:
    """generate random DNA sequence with given composition"""
    if composition is None:
        composition = {'A': 0.25, 'T': 0.25, 'G': 0.25, 'C': 0.25}

    bases = list(composition.keys())
    probs = list(composition.values())

    return ''.join(np.random.choice(bases, length, p=probs))


def insert_gene_patterns(sequence: str, island_type: str) -> str:
    """
    Insert simple gene-like patterns into island sequence

    This is very basic - just inserts start/stop codons and ORF-like patterns
    """
    seq_list = list(sequence)
    seq_len = len(seq_list)

    # add a few start codons
    start_codons = ['ATG', 'GTG', 'TTG']
    stop_codons = ['TAA', 'TAG', 'TGA']

    n_genes = random.randint(1, max(1, seq_len // 1000))  # roughly 1 gene per kb

    for _ in range(n_genes):
        # pick random position for start codon
        start_pos = random.randint(0, seq_len - 100)

        # insert start codon
        start_codon = random.choice(start_codons)
        for i, base in enumerate(start_codon):
            if start_pos + i < seq_len:
                seq_list[start_pos + i] = base

        # add stop codon downstream
        gene_length = random.randint(200, 1500)  # typical gene size
        stop_pos = start_pos + gene_length

        if stop_pos + 2 < seq_len:
            stop_codon = random.choice(stop_codons)
            for i, base in enumerate(stop_codon):
                if stop_pos + i < seq_len:
                    seq_list[stop_pos + i] = base

    return ''.join(seq_list)


def save_island_data(
    sequence_data: Dict[str, any],
    output_prefix: str,
    save_fasta: bool = True,
    save_annotations: bool = True,
):
    """
    Save the modified sequences and island annotations

    Args:
        sequence_data: output from read_and_insert_islands
        output_prefix: prefix for output files
        save_fasta: whether to save sequences as FASTA
        save_annotations: whether to save island positions as JSON
    """
    if save_fasta:
        from .io_handlers import save_fasta_sequences

        # save modified sequences
        modified_seqs = {
            seq_id: data['modified_sequence'] for seq_id, data in sequence_data.items()
        }
        save_fasta_sequences(modified_seqs, f'{output_prefix}_with_islands.fasta')

        # save original sequences too
        original_seqs = {
            seq_id: data['original_sequence'] for seq_id, data in sequence_data.items()
        }
        save_fasta_sequences(original_seqs, f'{output_prefix}_original.fasta')

    if save_annotations:
        import json

        # prepare annotation data
        annotations = {}
        for seq_id, data in sequence_data.items():
            annotations[seq_id] = {
                'original_length': data['original_length'],
                'modified_length': data['modified_length'],
                'islands': data['islands'],
            }

        with open(f'{output_prefix}_annotations.json', 'w') as f:
            json.dump(annotations, f, indent=2)

    print(f'Saved data with prefix: {output_prefix}')
