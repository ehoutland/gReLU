"""
Variant effect prediction for gene expression.

This module provides utilities to predict how genetic variants affect
gene expression using ridge regression models trained on epigenetic features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import h5py
from tqdm import tqdm
import warnings

from grelu.model.ridge import MultiGroupRidgeModel
from grelu.model.epigenetic_feature_extractor import (
    extract_features_for_variants,
    EpigeneticFeatureExtractor
)


def predict_variant_effects(
    model: MultiGroupRidgeModel,
    variant_features_dir: str,
    variants_df: pd.DataFrame,
    use_diff: bool = True,
    ref_col: str = 'ref',
    alt_col: str = 'alt',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Predict expression effects for variants using pre-computed features.

    Args:
        model: Trained MultiGroupRidgeModel
        variant_features_dir: Directory containing variant feature HDF5 files
        variants_df: DataFrame with variant information
        use_diff: If True, return alt-ref differences. If False, return ratios.
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        verbose: Show progress

    Returns:
        Tuple of (ref_predictions, alt_predictions, effects) where each is a
        DataFrame with variants as rows and cell types as columns
    """
    ref_preds_list = []
    alt_preds_list = []
    variant_ids = []
    problems = []

    iterator = variants_df.iterrows()
    if verbose:
        iterator = tqdm(list(iterator), desc="Predicting variant effects")

    for idx, variant in iterator:
        gene_name = variant['gene_name']
        tss = int(variant.get('tss', variant.get('start', 0)))

        # Load variant features
        feature_file = Path(variant_features_dir) / f"condensed_{gene_name}_{tss}.hd5f"

        if not feature_file.exists():
            problems.append(variant)
            continue

        try:
            with h5py.File(feature_file, 'r') as f:
                # Get reference features
                ref_features = f['full_tss'][:]

                # Get variant features
                var_pos = int(variant['pos'])
                alt_allele = variant[alt_col]
                mut_name = f"mut_{var_pos}_{alt_allele}"

                if mut_name not in f:
                    problems.append(variant)
                    continue

                alt_features = f[mut_name][:]

        except Exception as e:
            warnings.warn(f"Error loading features for {gene_name}: {e}")
            problems.append(variant)
            continue

        # Predict expression for ref and alt
        ref_pred = model.predict(ref_features.reshape(1, -1), return_dict=True)
        alt_pred = model.predict(alt_features.reshape(1, -1), return_dict=True)

        # Convert to arrays
        ref_pred_arr = np.array([ref_pred[g][0] for g in model.group_names])
        alt_pred_arr = np.array([alt_pred[g][0] for g in model.group_names])

        ref_preds_list.append(ref_pred_arr)
        alt_preds_list.append(alt_pred_arr)
        variant_ids.append(variant.get('SNP', f"{variant['chrom']}:{variant['pos']}"))

    if len(ref_preds_list) == 0:
        raise ValueError("No valid predictions could be made")

    # Convert to DataFrames
    ref_df = pd.DataFrame(
        np.vstack(ref_preds_list),
        columns=model.group_names,
        index=variant_ids
    )

    alt_df = pd.DataFrame(
        np.vstack(alt_preds_list),
        columns=model.group_names,
        index=variant_ids
    )

    # Compute effects
    if use_diff:
        effects_df = alt_df - ref_df
    else:
        effects_df = alt_df / (ref_df + 1e-10)  # Avoid division by zero

    if verbose and len(problems) > 0:
        print(f"Warning: {len(problems)} variants could not be processed")

    return ref_df, alt_df, effects_df


def predict_variant_effects_on_the_fly(
    model: MultiGroupRidgeModel,
    grelu_model,
    variants_df: pd.DataFrame,
    genome_fasta: str,
    seq_len: int = 20000,
    batch_size: int = 32,
    aggregation: str = 'mean',
    use_diff: bool = True,
    ref_col: str = 'ref',
    alt_col: str = 'alt',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Predict variant effects by computing epigenetic features on-the-fly.

    This function extracts features from the gReLU model directly without
    pre-computed features.

    Args:
        model: Trained MultiGroupRidgeModel
        grelu_model: Trained gReLU model (torch.nn.Module)
        variants_df: DataFrame with variant information
        genome_fasta: Path to genome FASTA file
        seq_len: Sequence length around variant
        batch_size: Batch size for feature extraction
        aggregation: How to aggregate epigenetic features
        use_diff: Return differences (True) or ratios (False)
        ref_col: Reference allele column
        alt_col: Alternate allele column
        verbose: Show progress

    Returns:
        Tuple of (ref_predictions, alt_predictions, effects)
    """
    if verbose:
        print("Extracting features for variants...")

    # Extract features for all variants
    variant_features = extract_features_for_variants(
        model=grelu_model,
        variants_df=variants_df,
        genome_fasta=genome_fasta,
        seq_len=seq_len,
        batch_size=batch_size,
        aggregation=aggregation,
        verbose=verbose
    )

    if verbose:
        print("Predicting expression effects...")

    ref_preds_list = []
    alt_preds_list = []
    variant_ids = []
    problems = []

    for idx, variant in tqdm(variants_df.iterrows(), total=len(variants_df), disable=not verbose):
        gene_name = variant['gene_name']

        if gene_name not in variant_features:
            problems.append(variant)
            continue

        gene_features = variant_features[gene_name]

        # Get reference features
        if 'full_tss' not in gene_features:
            problems.append(variant)
            continue

        ref_features = gene_features['full_tss']

        # Get variant features
        var_pos = int(variant['pos'])
        alt_allele = variant[alt_col]
        mut_name = f"mut_{var_pos}_{alt_allele}"

        if mut_name not in gene_features:
            problems.append(variant)
            continue

        alt_features = gene_features[mut_name]

        # Predict
        ref_pred = model.predict(ref_features.reshape(1, -1), return_dict=True)
        alt_pred = model.predict(alt_features.reshape(1, -1), return_dict=True)

        ref_pred_arr = np.array([ref_pred[g][0] for g in model.group_names])
        alt_pred_arr = np.array([alt_pred[g][0] for g in model.group_names])

        ref_preds_list.append(ref_pred_arr)
        alt_preds_list.append(alt_pred_arr)
        variant_ids.append(variant.get('SNP', f"{variant['chrom']}:{variant['pos']}"))

    if len(ref_preds_list) == 0:
        raise ValueError("No valid predictions could be made")

    # Convert to DataFrames
    ref_df = pd.DataFrame(
        np.vstack(ref_preds_list),
        columns=model.group_names,
        index=variant_ids
    )

    alt_df = pd.DataFrame(
        np.vstack(alt_preds_list),
        columns=model.group_names,
        index=variant_ids
    )

    # Compute effects
    if use_diff:
        effects_df = alt_df - ref_df
    else:
        effects_df = alt_df / (ref_df + 1e-10)

    if verbose and len(problems) > 0:
        print(f"Warning: {len(problems)} variants could not be processed")

    return ref_df, alt_df, effects_df


def compute_delta_representations(
    variant_features_dir: str,
    variants_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute the L2 norm difference between reference and variant epigenetic features.

    This can be used as a metric of how much a variant perturbs the epigenetic
    landscape.

    Args:
        variant_features_dir: Directory with variant feature HDF5 files
        variants_df: DataFrame with variant information
        verbose: Show progress

    Returns:
        DataFrame with variant IDs and delta values
    """
    delta_dict = {}

    iterator = variants_df.iterrows()
    if verbose:
        iterator = tqdm(list(iterator), desc="Computing deltas")

    for idx, variant in iterator:
        gene_name = variant['gene_name']
        tss = int(variant.get('tss', variant.get('start', 0)))

        feature_file = Path(variant_features_dir) / f"condensed_{gene_name}_{tss}.hd5f"

        if not feature_file.exists():
            continue

        try:
            with h5py.File(feature_file, 'r') as f:
                ref_features = f['full_tss'][:]

                var_pos = int(variant['pos'])
                alt_allele = variant['alt']
                mut_name = f"mut_{var_pos}_{alt_allele}"

                if mut_name not in f:
                    continue

                alt_features = f[mut_name][:]

                # Compute L2 norm
                delta = np.linalg.norm(alt_features - ref_features)

                variant_id = variant.get('SNP', f"{variant['chrom']}:{variant['pos']}")
                delta_dict[variant_id] = delta

        except Exception:
            continue

    delta_df = pd.DataFrame.from_dict(
        delta_dict,
        orient='index',
        columns=['delta_epigenetic']
    )

    return delta_df


def save_variant_effects(
    ref_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    delta_df: Optional[pd.DataFrame],
    output_dir: str,
    prefix: str = ''
):
    """
    Save variant effect predictions to files.

    Args:
        ref_df: Reference allele predictions
        alt_df: Alternate allele predictions
        effects_df: Effect sizes (alt - ref or alt / ref)
        delta_df: Optional delta epigenetic features
        output_dir: Output directory
        prefix: Prefix for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save intermediate files
    intermediate_dir = output_path / 'intermediate'
    intermediate_dir.mkdir(exist_ok=True)

    ref_df.to_csv(intermediate_dir / f'{prefix}REF.csv')
    alt_df.to_csv(intermediate_dir / f'{prefix}ALT.csv')

    if delta_df is not None:
        delta_df.to_csv(intermediate_dir / f'{prefix}deltaEpigenetic.csv')

    # Save final effects
    effects_df.to_csv(output_path / f'{prefix}effects.csv')


def get_top_effects_per_cell_type(
    effects_df: pd.DataFrame,
    n_top: int = 100,
    use_abs: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Get top variant effects for each cell type.

    Args:
        effects_df: DataFrame with variants as rows, cell types as columns
        n_top: Number of top variants to return
        use_abs: Use absolute values for ranking

    Returns:
        Dictionary mapping cell type to DataFrame of top variants
    """
    top_variants = {}

    for cell_type in effects_df.columns:
        effects = effects_df[cell_type]

        if use_abs:
            effects_sorted = effects.abs().sort_values(ascending=False)
        else:
            effects_sorted = effects.sort_values(ascending=False)

        top_idx = effects_sorted.head(n_top).index
        top_variants[cell_type] = effects_df.loc[top_idx, [cell_type]]

    return top_variants


def annotate_variants_with_effects(
    variants_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    effect_threshold: Optional[float] = None,
    max_effect_column: bool = True
) -> pd.DataFrame:
    """
    Annotate variant DataFrame with expression effects.

    Args:
        variants_df: Original variant DataFrame
        effects_df: Effect predictions
        effect_threshold: Optional threshold for filtering
        max_effect_column: Add column with cell type showing max effect

    Returns:
        Annotated variant DataFrame
    """
    # Merge with effects
    variants_annotated = variants_df.copy()

    # Add SNP ID if not present
    if 'SNP' not in variants_annotated.columns:
        variants_annotated['SNP'] = (
            variants_annotated['chrom'].astype(str) + ':' +
            variants_annotated['pos'].astype(str)
        )

    variants_annotated = variants_annotated.set_index('SNP')
    variants_annotated = variants_annotated.join(effects_df, how='left')

    # Add max effect column
    if max_effect_column:
        effect_cols = effects_df.columns
        variants_annotated['max_effect_cell_type'] = (
            variants_annotated[effect_cols].abs().idxmax(axis=1)
        )
        variants_annotated['max_effect_value'] = (
            variants_annotated[effect_cols].abs().max(axis=1)
        )

    # Filter by threshold if provided
    if effect_threshold is not None:
        mask = variants_annotated['max_effect_value'] >= effect_threshold
        variants_annotated = variants_annotated[mask]

    return variants_annotated.reset_index()
