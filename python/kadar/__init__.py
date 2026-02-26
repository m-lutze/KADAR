__version__ = "1.0.0"

# Import main classes
from .core.kmer_profiler import KmerProfiler
from .core.predictor import GenomeIslandPredictor
from .utils.io_handlers import load_fasta_sequences
from .utils.synthetic_data import read_and_insert_islands

__all__ = [
    'KmerProfiler',
    'GenomeIslandPredictor',
    'load_fasta_sequences',
    'read_and_insert_islands'
]