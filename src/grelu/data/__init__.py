"""
`gReLU.data` contains modules related to processing and QC of genomic data, and loading
and augmenting genomic data for training, validation and testing sequence-to-function models.
Also includes utilities for handling single-cell expression data.
"""

# Single-cell expression data utilities
from grelu.data.single_cell import (
    load_expression_matrix,
    aggregate_expression_by_cell_type,
    prepare_data_matrices,
    split_genes_by_chromosome,
    get_gene_weights,
    check_gene_existence,
    SingleCellExpressionDataset,
    filter_cell_types
)

__all__ = [
    'load_expression_matrix',
    'aggregate_expression_by_cell_type',
    'prepare_data_matrices',
    'split_genes_by_chromosome',
    'get_gene_weights',
    'check_gene_existence',
    'SingleCellExpressionDataset',
    'filter_cell_types',
]
