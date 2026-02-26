# KADAR

A Python package for predicting genome islands (horizontally transferred genomic
regions) based on compositional characteristics and k-mer analysis of DNA
sequences.

Testing FASTA and notebook included.

## Installation

```bash
# develop mode
pip install maturin
maturin develop --extras dev,docs

# OR build release
maturin build --release
```