"""
Single-cell expression data utilities for ridge regression training.

This module handles single-cell expression data and prepares it for training
ridge regression models that map epigenetic features to gene expression.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import warnings


def load_expression_matrix(
    file_path: str,
    transpose: bool = False
) -> pd.DataFrame:
    """
    Load expression matrix from file.

    Args:
        file_path: Path to expression matrix (CSV, TSV, or pickle)
        transpose: If True, transpose so genes are columns and cells/groups are rows

    Returns:
        DataFrame with genes as columns and cells/cell types as rows (after transpose if needed)
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == '.pkl' or suffix == '.pickle':
        expr_df = pd.read_pickle(file_path)
    elif suffix == '.csv':
        expr_df = pd.read_csv(file_path, index_col=0)
    elif suffix == '.tsv' or suffix == '.txt':
        expr_df = pd.read_csv(file_path, sep='\t', index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if transpose:
        expr_df = expr_df.T

    return expr_df


def aggregate_expression_by_cell_type(
    expression_matrix: pd.DataFrame,
    cell_annotations: Union[pd.DataFrame, pd.Series, Dict],
    cell_type_column: str = 'cell_type',
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Aggregate single-cell expression by cell type.

    Args:
        expression_matrix: DataFrame with genes as columns, cells as rows
        cell_annotations: DataFrame/Series/Dict mapping cells to cell types
        cell_type_column: Column name in cell_annotations with cell type labels
        aggregation: Aggregation method ('mean', 'median', 'sum')

    Returns:
        DataFrame with genes as columns, cell types as rows
    """
    # Convert cell_annotations to Series if needed
    if isinstance(cell_annotations, dict):
        cell_annotations = pd.Series(cell_annotations)
    elif isinstance(cell_annotations, pd.DataFrame):
        cell_annotations = cell_annotations[cell_type_column]

    # Align cells
    common_cells = expression_matrix.index.intersection(cell_annotations.index)
    if len(common_cells) == 0:
        raise ValueError("No common cells between expression matrix and annotations")

    expression_matrix = expression_matrix.loc[common_cells]
    cell_annotations = cell_annotations.loc[common_cells]

    # Add cell type labels
    expression_matrix['cell_type'] = cell_annotations

    # Aggregate
    if aggregation == 'mean':
        aggregated = expression_matrix.groupby('cell_type').mean()
    elif aggregation == 'median':
        aggregated = expression_matrix.groupby('cell_type').median()
    elif aggregation == 'sum':
        aggregated = expression_matrix.groupby('cell_type').sum()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return aggregated


def prepare_data_matrices(
    genes_df: pd.DataFrame,
    epigenetic_features: Dict[str, np.ndarray],
    expression_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare aligned data matrices for training.

    Args:
        genes_df: DataFrame with 'gene_name' column specifying genes to use
        epigenetic_features: Dictionary mapping gene names to feature vectors
        expression_df: DataFrame with genes as columns, cell types as rows

    Returns:
        Tuple of (dataX, dataY, group_names) where:
            dataX: (n_genes, n_features) array of epigenetic features
            dataY: (n_genes, n_groups) array of expression values
            group_names: Array of group/cell type names
    """
    n_genes = len(genes_df)
    group_names = expression_df.index.values

    # Get feature dimension from first gene
    first_gene = genes_df.iloc[0]['gene_name']
    if first_gene not in epigenetic_features:
        raise ValueError(f"Gene {first_gene} not found in epigenetic_features")
    n_features = len(epigenetic_features[first_gene])

    # Initialize arrays
    dataX = np.zeros((n_genes, n_features))
    dataY = np.zeros((n_genes, len(group_names)))

    # Fill arrays
    for i, (_, gene_row) in enumerate(genes_df.iterrows()):
        gene_name = gene_row['gene_name']

        # Get epigenetic features
        if gene_name not in epigenetic_features:
            warnings.warn(f"Gene {gene_name} not in epigenetic_features, using zeros")
            continue
        dataX[i, :] = epigenetic_features[gene_name]

        # Get expression
        if gene_name not in expression_df.columns:
            warnings.warn(f"Gene {gene_name} not in expression_df, using zeros")
            continue
        dataY[i, :] = expression_df[gene_name].values

    return dataX, dataY, group_names


def split_genes_by_chromosome(
    genes_df: pd.DataFrame,
    test_chroms: List[str] = ['8', '9'],
    val_fraction: float = 0.1,
    random_state: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split genes into train, validation, and test sets by chromosome.

    Args:
        genes_df: DataFrame with 'chrom' and 'gene_name' columns
        test_chroms: Chromosomes to hold out for testing
        val_fraction: Fraction of remaining data for validation
        random_state: Random seed

    Returns:
        Tuple of (train_genes, val_genes, test_genes)
    """
    # Remove sex chromosomes
    genes_df = genes_df[~genes_df['chrom'].isin(['X', 'Y', 'chrX', 'chrY'])]

    # Normalize chromosome names
    genes_df = genes_df.copy()
    genes_df['chrom'] = genes_df['chrom'].astype(str).str.replace('chr', '')

    # Test set from specified chromosomes
    test_genes = genes_df[genes_df['chrom'].isin(test_chroms)]

    # Train/val from remaining chromosomes
    train_val_genes = genes_df[~genes_df['chrom'].isin(test_chroms)]

    # Split train/val
    if val_fraction > 0:
        train_genes, val_genes = train_test_split(
            train_val_genes,
            test_size=val_fraction,
            random_state=random_state
        )
    else:
        train_genes = train_val_genes
        val_genes = pd.DataFrame(columns=genes_df.columns)

    return train_genes, val_genes, test_genes


def get_gene_weights(
    expression_array: np.ndarray,
    percentile_cutoff_top: float = 70,
    percentile_cutoff_bottom: float = 10
) -> np.ndarray:
    """
    Assign weights to genes based on expression variance across groups.

    Genes with higher variance get higher weights. This helps the model
    focus on genes with more dynamic expression patterns.

    Args:
        expression_array: (n_genes, n_groups) array of expression values
        percentile_cutoff_top: Percentile threshold for high variance genes
        percentile_cutoff_bottom: Percentile threshold for low variance genes

    Returns:
        Array of shape (n_genes,) with weights for each gene
    """
    # Center expression
    expression_centered = expression_array - expression_array.mean(axis=1, keepdims=True)

    # Compute standard deviation across groups
    gene_std = expression_centered.std(axis=1)

    # Initialize weights
    weights = np.ones(len(gene_std))

    # Increase weight for high variance genes
    top_threshold = np.percentile(gene_std, percentile_cutoff_top)
    weights[gene_std >= top_threshold] += 1

    # Decrease weight for low variance genes
    bottom_threshold = np.percentile(gene_std, percentile_cutoff_bottom)
    weights[gene_std < bottom_threshold] -= 0.5

    return weights


def check_gene_existence(
    genes_df: pd.DataFrame,
    epigenetic_features: Dict[str, np.ndarray],
    expression_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter genes to only those present in both features and expression data.

    Args:
        genes_df: DataFrame with 'gene_name' column
        epigenetic_features: Dictionary of gene features
        expression_df: DataFrame with genes as columns

    Returns:
        Filtered genes DataFrame
    """
    # Get genes present in features
    genes_in_features = set(epigenetic_features.keys())
    genes_df = genes_df[genes_df['gene_name'].isin(genes_in_features)]

    # Get genes present in expression
    genes_in_expression = set(expression_df.columns)
    genes_df = genes_df[genes_df['gene_name'].isin(genes_in_expression)]

    return genes_df


class SingleCellExpressionDataset:
    """
    Dataset class for single-cell expression data with epigenetic features.

    This class handles the pairing of epigenetic features with expression data
    and provides utilities for train/val/test splitting.

    Args:
        genes_df: DataFrame with gene information (gene_name, chrom, tss, etc.)
        epigenetic_features: Dictionary mapping gene names to feature arrays
        expression_df: DataFrame with genes as columns, groups/cell types as rows
        test_chroms: Chromosomes to use for testing
        val_fraction: Fraction of training data to use for validation
        random_state: Random seed for splitting
        check_existence: Whether to filter genes present in both features and expression

    Attributes:
        train_genes: Training gene names
        val_genes: Validation gene names
        test_genes: Test gene names
        group_names: Cell type/group names
        dataX_train, dataY_train: Training data
        dataX_val, dataY_val: Validation data
        dataX_test, dataY_test: Test data
    """

    def __init__(
        self,
        genes_df: pd.DataFrame,
        epigenetic_features: Dict[str, np.ndarray],
        expression_df: pd.DataFrame,
        test_chroms: List[str] = ['8', '9'],
        val_fraction: float = 0.1,
        random_state: int = 3,
        check_existence: bool = True
    ):
        self.genes_df = genes_df.copy()
        self.epigenetic_features = epigenetic_features
        self.expression_df = expression_df
        self.test_chroms = test_chroms
        self.val_fraction = val_fraction
        self.random_state = random_state

        # Filter genes if requested
        if check_existence:
            self.genes_df = check_gene_existence(
                self.genes_df, epigenetic_features, expression_df
            )

        # Split genes
        self.train_genes_df, self.val_genes_df, self.test_genes_df = \
            split_genes_by_chromosome(
                self.genes_df, test_chroms, val_fraction, random_state
            )

        # Prepare data matrices
        self.dataX_train, self.dataY_train, self.group_names = prepare_data_matrices(
            self.train_genes_df, epigenetic_features, expression_df
        )

        if len(self.val_genes_df) > 0:
            self.dataX_val, self.dataY_val, _ = prepare_data_matrices(
                self.val_genes_df, epigenetic_features, expression_df
            )
        else:
            self.dataX_val = np.array([])
            self.dataY_val = np.array([])

        if len(self.test_genes_df) > 0:
            self.dataX_test, self.dataY_test, _ = prepare_data_matrices(
                self.test_genes_df, epigenetic_features, expression_df
            )
        else:
            self.dataX_test = np.array([])
            self.dataY_test = np.array([])

    @property
    def train_genes(self) -> np.ndarray:
        """Get training gene names."""
        return self.train_genes_df['gene_name'].values

    @property
    def val_genes(self) -> np.ndarray:
        """Get validation gene names."""
        return self.val_genes_df['gene_name'].values if len(self.val_genes_df) > 0 else np.array([])

    @property
    def test_genes(self) -> np.ndarray:
        """Get test gene names."""
        return self.test_genes_df['gene_name'].values if len(self.test_genes_df) > 0 else np.array([])

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data (X, Y)."""
        return self.dataX_train, self.dataY_train

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data (X, Y)."""
        return self.dataX_val, self.dataY_val

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data (X, Y)."""
        return self.dataX_test, self.dataY_test

    def save(self, output_path: str):
        """Save dataset to pickle file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'train': {
                    'X': self.dataX_train,
                    'Y': self.dataY_train,
                    'genes': self.train_genes,
                    'groups': self.group_names
                },
                'val': {
                    'X': self.dataX_val,
                    'Y': self.dataY_val,
                    'genes': self.val_genes,
                    'groups': self.group_names
                },
                'test': {
                    'X': self.dataX_test,
                    'Y': self.dataY_test,
                    'genes': self.test_genes,
                    'groups': self.group_names
                }
            }, f)

    @classmethod
    def load(cls, input_path: str):
        """Load dataset from pickle file."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        # Create a dummy instance and populate it
        # This is a simplified loader - full reconstruction would need all original params
        return data


def filter_cell_types(
    expression_df: pd.DataFrame,
    remove_immune: bool = True,
    remove_ambiguous: bool = True
) -> pd.DataFrame:
    """
    Filter cell types based on immune and ambiguous criteria.

    Args:
        expression_df: DataFrame with cell types as rows, genes as columns
        remove_immune: Remove immune cell types
        remove_ambiguous: Remove ambiguous/unannotated cell types

    Returns:
        Filtered DataFrame
    """
    cell_types = expression_df.index.tolist()
    keep_types = []

    immune_keywords = [
        'bcell', 'b_cell', 'inflammatorymacs', 'nklike', 'abtcell', 'gdtcell',
        'cd4tcell', 'cd8tcell', 't_cd4', 't_cd8', 't_r', 'monocytederived',
        'mnpcdendriticcell', 'nktcell', 'nkcell', 'dc_', 'macrophage',
        'monocyte', 'nk_', 'mast_', 'lymphoid'
    ]

    ambiguous_keywords = [
        'unannotated', 'unspecified', 'notassigned', 'unclassified',
        'doublet', 'unknown'
    ]

    for cell_type in cell_types:
        cell_type_lower = cell_type.lower().replace(' ', '').replace('-', '').replace('_', '')

        # Check immune
        is_immune = any(keyword in cell_type_lower for keyword in immune_keywords)
        if remove_immune and is_immune:
            continue

        # Check ambiguous
        is_ambiguous = any(keyword in cell_type_lower for keyword in ambiguous_keywords)
        if remove_ambiguous and is_ambiguous:
            continue

        keep_types.append(cell_type)

    return expression_df.loc[keep_types]
