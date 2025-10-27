"""
Feature extraction from gReLU models for downstream ridge regression.

This module extracts epigenetic predictions from trained gReLU models to use
as features for predicting gene expression with ridge regression models.
"""

import numpy as np
import pandas as pd
import torch
from typing import Union, Optional, Callable, Dict, List
from pathlib import Path
import pickle
import h5py
from tqdm import tqdm

from grelu.sequence.format import strings_to_one_hot
from grelu.io.genome import read_fasta


class EpigeneticFeatureExtractor:
    """
    Extract epigenetic features from a trained gReLU model.

    This class takes a trained gReLU model and extracts predictions for genes,
    which can then be used as features for ridge regression to predict expression.

    Args:
        model: Trained gReLU model (torch.nn.Module or LightningModule)
        genome_fasta: Path to genome FASTA file for extracting sequences
        layer_name: Optional name of intermediate layer to extract features from.
                   If None, uses final model output.
        aggregation: How to aggregate across sequence length. Options:
                    'mean', 'max', 'sum', or a callable function.
        device: Device to run model on ('cuda' or 'cpu')

    Example:
        >>> extractor = EpigeneticFeatureExtractor(
        ...     model=trained_model,
        ...     genome_fasta='hg38.fa',
        ...     aggregation='mean'
        ... )
        >>> features = extractor.extract_features(genes_df)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        genome_fasta: Optional[str] = None,
        layer_name: Optional[str] = None,
        aggregation: Union[str, Callable] = 'mean',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.genome_fasta = genome_fasta
        self.layer_name = layer_name
        self.device = device

        # Set up aggregation function
        if aggregation == 'mean':
            self.aggregation = lambda x: x.mean(dim=-1)
        elif aggregation == 'max':
            self.aggregation = lambda x: x.max(dim=-1)[0]
        elif aggregation == 'sum':
            self.aggregation = lambda x: x.sum(dim=-1)
        elif callable(aggregation):
            self.aggregation = aggregation
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Hook for intermediate layer extraction
        self.activation = {}
        if layer_name:
            self._register_hook(layer_name)

    def _register_hook(self, layer_name: str):
        """Register forward hook to extract intermediate layer activations."""
        def hook(module, input, output):
            self.activation[layer_name] = output

        # Find and register hook on the specified layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook)
                return
        raise ValueError(f"Layer {layer_name} not found in model")

    def extract_features_from_sequences(
        self,
        sequences: Union[List[str], np.ndarray],
        batch_size: int = 32,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Extract features from DNA sequences.

        Args:
            sequences: List of DNA sequences or one-hot encoded array
            batch_size: Batch size for inference
            verbose: Show progress bar

        Returns:
            Array of shape (n_sequences, n_features) containing epigenetic features
        """
        # Convert sequences to one-hot if needed
        if isinstance(sequences, list):
            sequences = strings_to_one_hot(sequences)

        n_sequences = len(sequences)
        features_list = []

        iterator = range(0, n_sequences, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Extracting features")

        with torch.no_grad():
            for i in iterator:
                batch = sequences[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

                # Forward pass
                output = self.model(batch_tensor)

                # Get features from specified layer or final output
                if self.layer_name:
                    features = self.activation[self.layer_name]
                else:
                    features = output

                # Aggregate across sequence length: (batch, tasks, length) -> (batch, tasks)
                if features.dim() == 3:
                    features = self.aggregation(features)

                # Flatten if needed: (batch, tasks) -> (batch, tasks)
                features = features.cpu().numpy()
                if features.ndim == 1:
                    features = features.reshape(-1, 1)

                features_list.append(features)

        return np.vstack(features_list)

    def extract_features_from_genes(
        self,
        genes_df: pd.DataFrame,
        genome_fasta: Optional[str] = None,
        seq_len: int = 20000,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for genes from their TSS regions.

        Args:
            genes_df: DataFrame with columns: 'gene_name', 'chrom', 'tss' (or 'start')
                     Optional: 'strand'
            genome_fasta: Path to genome FASTA. Uses self.genome_fasta if not provided.
            seq_len: Length of sequence to extract around TSS
            batch_size: Batch size for inference
            verbose: Show progress bar

        Returns:
            Dictionary mapping gene names to feature vectors
        """
        if genome_fasta is None:
            genome_fasta = self.genome_fasta
        if genome_fasta is None:
            raise ValueError("genome_fasta must be provided")

        # Determine TSS position column
        tss_col = 'tss' if 'tss' in genes_df.columns else 'start'

        # Extract sequences around TSS
        sequences = []
        gene_names = []

        if verbose:
            print("Extracting sequences from genome...")

        for _, gene in genes_df.iterrows():
            chrom = gene['chrom'] if 'chr' in str(gene['chrom']) else f"chr{gene['chrom']}"
            tss = int(gene[tss_col])
            start = max(0, tss - seq_len // 2)
            end = start + seq_len

            # Read sequence
            seq = read_fasta(genome_fasta, chrom, start, end)

            # Reverse complement if on minus strand
            if 'strand' in gene and gene['strand'] == '-':
                seq = self._reverse_complement(seq)

            sequences.append(seq)
            gene_names.append(gene['gene_name'])

        # Extract features
        features = self.extract_features_from_sequences(
            sequences, batch_size=batch_size, verbose=verbose
        )

        # Create dictionary
        return {name: feat for name, feat in zip(gene_names, features)}

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Reverse complement a DNA sequence."""
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))

    def save_features(
        self,
        features_dict: Dict[str, np.ndarray],
        output_path: str,
        format: str = 'pickle'
    ):
        """
        Save extracted features to file.

        Args:
            features_dict: Dictionary mapping gene names to features
            output_path: Path to save file
            format: 'pickle' or 'hdf5'
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(features_dict, f)
        elif format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                for gene_name, features in features_dict.items():
                    f.create_dataset(gene_name, data=features)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def load_features(
        input_path: str,
        format: str = 'pickle'
    ) -> Dict[str, np.ndarray]:
        """
        Load features from file.

        Args:
            input_path: Path to features file
            format: 'pickle' or 'hdf5'

        Returns:
            Dictionary mapping gene names to features
        """
        if format == 'pickle':
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        elif format == 'hdf5':
            features_dict = {}
            with h5py.File(input_path, 'r') as f:
                for gene_name in f.keys():
                    features_dict[gene_name] = f[gene_name][:]
            return features_dict
        else:
            raise ValueError(f"Unknown format: {format}")


def extract_features_for_variants(
    model: torch.nn.Module,
    variants_df: pd.DataFrame,
    genome_fasta: str,
    seq_len: int = 20000,
    batch_size: int = 32,
    aggregation: str = 'mean',
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract features for both reference and variant alleles.

    Args:
        model: Trained gReLU model
        variants_df: DataFrame with columns: 'chrom', 'pos', 'ref', 'alt', 'gene_name', 'tss'
        genome_fasta: Path to genome FASTA
        seq_len: Sequence length
        batch_size: Batch size
        aggregation: Aggregation method
        save_dir: Optional directory to save individual variant features
        verbose: Show progress

    Returns:
        Dictionary with structure:
        {
            'gene_name': {
                'full_tss': array,
                'mut_pos_allele': array,
                ...
            }
        }
    """
    extractor = EpigeneticFeatureExtractor(
        model=model,
        genome_fasta=genome_fasta,
        aggregation=aggregation
    )

    variant_features = {}

    # Group variants by gene
    for gene_name, gene_variants in variants_df.groupby('gene_name'):
        gene_features = {}

        # Get reference sequence
        gene_info = gene_variants.iloc[0]
        chrom = gene_info['chrom'] if 'chr' in str(gene_info['chrom']) else f"chr{gene_info['chrom']}"
        tss = int(gene_info['tss'])
        start = max(0, tss - seq_len // 2)
        end = start + seq_len

        ref_seq = read_fasta(genome_fasta, chrom, start, end)

        if 'strand' in gene_info and gene_info['strand'] == '-':
            ref_seq = extractor._reverse_complement(ref_seq)

        # Extract reference features
        ref_features = extractor.extract_features_from_sequences([ref_seq], batch_size=1, verbose=False)
        gene_features['full_tss'] = ref_features[0]

        # Extract variant features
        for _, variant in gene_variants.iterrows():
            var_pos = int(variant['pos']) - start
            if 0 <= var_pos < len(ref_seq):
                # Create mutated sequence
                mut_seq = list(ref_seq)
                mut_seq[var_pos] = variant['alt']
                mut_seq = ''.join(mut_seq)

                # Extract features
                mut_features = extractor.extract_features_from_sequences([mut_seq], batch_size=1, verbose=False)
                mut_name = f"mut_{variant['pos']}_{variant['alt']}"
                gene_features[mut_name] = mut_features[0]

        variant_features[gene_name] = gene_features

        # Optionally save to HDF5
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f"condensed_{gene_name}_{tss}.hd5f"
            with h5py.File(save_path, 'w') as f:
                for name, features in gene_features.items():
                    f.create_dataset(name, data=features)

    return variant_features
