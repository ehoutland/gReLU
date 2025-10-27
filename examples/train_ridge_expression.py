"""
Example script for training ridge regression models to predict gene expression.

This script demonstrates the complete workflow:
1. Load a trained gReLU model
2. Extract epigenetic features for genes
3. Load single-cell expression data
4. Train ridge regression models (one per cell type)
5. Evaluate on validation and test sets
6. Predict variant effects on expression

Usage:
    python train_ridge_expression.py --help
"""

import argparse
import pandas as pd
import torch
from pathlib import Path

from grelu.lightning import LightningModel
from grelu.model import (
    EpigeneticFeatureExtractor,
    train_expression_models,
    tune_and_train,
    cross_validate_expression_models,
    evaluate_on_test_set,
    predict_variant_effects_on_the_fly
)
from grelu.data import load_expression_matrix, filter_cell_types


def main(args):
    """Main training pipeline."""

    print("=" * 80)
    print("Training Ridge Regression Models for Gene Expression Prediction")
    print("=" * 80)

    # =========================================================================
    # Step 1: Load trained gReLU model
    # =========================================================================
    print("\n[1/6] Loading gReLU model...")

    if args.grelu_model.endswith('.ckpt'):
        # Load PyTorch Lightning checkpoint
        grelu_model = LightningModel.load_from_checkpoint(args.grelu_model)
        grelu_model = grelu_model.model
    else:
        # Load PyTorch model
        grelu_model = torch.load(args.grelu_model)

    grelu_model.eval()
    print(f"   Loaded model from: {args.grelu_model}")

    # =========================================================================
    # Step 2: Extract epigenetic features for genes
    # =========================================================================
    print("\n[2/6] Extracting epigenetic features...")

    # Load gene list
    genes_df = pd.read_csv(args.genes_file)
    print(f"   Loaded {len(genes_df)} genes from: {args.genes_file}")

    # Check if features already exist
    features_file = args.output_dir / 'epigenetic_features.pkl'

    if features_file.exists() and not args.recompute_features:
        print(f"   Loading pre-computed features from: {features_file}")
        epigenetic_features = EpigeneticFeatureExtractor.load_features(
            str(features_file)
        )
    else:
        # Extract features from gReLU model
        print(f"   Extracting features from sequences...")
        extractor = EpigeneticFeatureExtractor(
            model=grelu_model,
            genome_fasta=args.genome_fasta,
            aggregation=args.aggregation,
            device=args.device
        )

        epigenetic_features = extractor.extract_features_from_genes(
            genes_df=genes_df,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            verbose=True
        )

        # Save features
        print(f"   Saving features to: {features_file}")
        extractor.save_features(epigenetic_features, str(features_file))

    print(f"   Feature dimension: {len(next(iter(epigenetic_features.values())))}")

    # =========================================================================
    # Step 3: Load expression data
    # =========================================================================
    print("\n[3/6] Loading expression data...")

    expression_df = load_expression_matrix(
        args.expression_file,
        transpose=args.transpose_expression
    )
    print(f"   Expression matrix shape: {expression_df.shape}")
    print(f"   (rows={expression_df.shape[0]} groups, cols={expression_df.shape[1]} genes)")

    # Filter cell types if requested
    if args.filter_immune or args.filter_ambiguous:
        print(f"   Filtering cell types...")
        expression_df = filter_cell_types(
            expression_df,
            remove_immune=args.filter_immune,
            remove_ambiguous=args.filter_ambiguous
        )
        print(f"   After filtering: {expression_df.shape[0]} cell types")

    # =========================================================================
    # Step 4: Train ridge regression models
    # =========================================================================
    print("\n[4/6] Training ridge regression models...")

    if args.tune_alpha:
        print("   Tuning alpha hyperparameter...")
        model, best_alpha, results = tune_and_train(
            genes_df=genes_df,
            epigenetic_features=epigenetic_features,
            expression_df=expression_df,
            alphas=args.alpha_values,
            cv_folds=args.cv_folds,
            test_chroms=args.test_chroms,
            use_log_transform=args.use_log_transform,
            pseudocount=args.pseudocount,
            use_weights=args.use_weights,
            save_models=True,
            model_save_dir=str(args.output_dir / 'models'),
            random_state=args.random_state,
            verbose=True
        )
        print(f"   Best alpha: {best_alpha}")

    else:
        print(f"   Training with alpha={args.alpha}...")
        model, results, preds_df, truth_df = train_expression_models(
            genes_df=genes_df,
            epigenetic_features=epigenetic_features,
            expression_df=expression_df,
            alpha=args.alpha,
            test_chroms=args.test_chroms,
            val_fraction=args.val_fraction,
            use_log_transform=args.use_log_transform,
            pseudocount=args.pseudocount,
            use_weights=args.use_weights,
            filter_immune=args.filter_immune,
            filter_ambiguous=args.filter_ambiguous,
            save_models=True,
            model_save_dir=str(args.output_dir / 'models'),
            save_predictions=True,
            predictions_save_dir=str(args.output_dir / 'predictions'),
            random_state=args.random_state,
            verbose=True
        )

    # Print validation results
    if results:
        print("\n   Validation Results (Spearman correlation):")
        for group, metrics in list(results.items())[:5]:  # Show first 5
            spearman = metrics.get('spearman', 'N/A')
            print(f"      {group}: {spearman:.4f}")
        if len(results) > 5:
            print(f"      ... and {len(results) - 5} more cell types")

    # =========================================================================
    # Step 5: Cross-validation (optional)
    # =========================================================================
    if args.run_cv:
        print("\n[5/6] Running cross-validation...")

        cv_results, predictions, summary_df = cross_validate_expression_models(
            genes_df=genes_df,
            epigenetic_features=epigenetic_features,
            expression_df=expression_df,
            alpha=args.alpha,
            n_folds=args.cv_folds,
            test_chroms=args.test_chroms,
            use_log_transform=args.use_log_transform,
            pseudocount=args.pseudocount,
            use_weights=args.use_weights,
            save_predictions=True,
            predictions_save_dir=str(args.output_dir / 'cv_results'),
            random_state=args.random_state,
            verbose=True
        )

        print("\n   Cross-validation Summary:")
        print(summary_df[['spearman_mean', 'spearman_std']].head())
    else:
        print("\n[5/6] Skipping cross-validation (use --run_cv to enable)")

    # =========================================================================
    # Step 6: Evaluate on test set
    # =========================================================================
    print("\n[6/6] Evaluating on test set...")

    test_results, test_preds, test_truth = evaluate_on_test_set(
        model=model,
        genes_df=genes_df,
        epigenetic_features=epigenetic_features,
        expression_df=expression_df,
        test_chroms=args.test_chroms,
        use_log_transform=args.use_log_transform,
        pseudocount=args.pseudocount,
        save_predictions=True,
        predictions_save_dir=str(args.output_dir / 'test_results'),
        verbose=True
    )

    print("\n   Test Results (Spearman correlation):")
    for group, metrics in list(test_results.items())[:5]:
        spearman = metrics.get('spearman', 'N/A')
        print(f"      {group}: {spearman:.4f}")

    # =========================================================================
    # Optional: Predict variant effects
    # =========================================================================
    if args.variants_file:
        print("\n[Optional] Predicting variant effects...")

        variants_df = pd.read_csv(args.variants_file)
        print(f"   Loaded {len(variants_df)} variants")

        ref_preds, alt_preds, effects = predict_variant_effects_on_the_fly(
            model=model,
            grelu_model=grelu_model,
            variants_df=variants_df,
            genome_fasta=args.genome_fasta,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            use_diff=True,
            verbose=True
        )

        # Save variant effects
        output_dir = args.output_dir / 'variant_effects'
        output_dir.mkdir(exist_ok=True)

        effects.to_csv(output_dir / 'variant_effects.csv')
        ref_preds.to_csv(output_dir / 'ref_predictions.csv')
        alt_preds.to_csv(output_dir / 'alt_predictions.csv')

        print(f"   Saved variant effects to: {output_dir}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ridge regression models for gene expression prediction'
    )

    # Input files
    parser.add_argument(
        '--grelu_model',
        type=str,
        required=True,
        help='Path to trained gReLU model (.ckpt or .pt)'
    )
    parser.add_argument(
        '--genes_file',
        type=str,
        required=True,
        help='CSV file with gene information (gene_name, chrom, tss)'
    )
    parser.add_argument(
        '--expression_file',
        type=str,
        required=True,
        help='Expression matrix file (CSV/TSV/pickle)'
    )
    parser.add_argument(
        '--genome_fasta',
        type=str,
        required=True,
        help='Path to genome FASTA file'
    )
    parser.add_argument(
        '--variants_file',
        type=str,
        default=None,
        help='Optional: CSV file with variants for effect prediction'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./ridge_output'),
        help='Output directory (default: ./ridge_output)'
    )

    # Feature extraction
    parser.add_argument(
        '--seq_len',
        type=int,
        default=20000,
        help='Sequence length around TSS (default: 20000)'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max', 'sum'],
        help='How to aggregate epigenetic features (default: mean)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for model (default: cuda if available)'
    )
    parser.add_argument(
        '--recompute_features',
        action='store_true',
        help='Recompute features even if they exist'
    )

    # Expression data
    parser.add_argument(
        '--transpose_expression',
        action='store_true',
        help='Transpose expression matrix (if genes are rows)'
    )
    parser.add_argument(
        '--filter_immune',
        action='store_true',
        help='Remove immune cell types'
    )
    parser.add_argument(
        '--filter_ambiguous',
        action='store_true',
        help='Remove ambiguous/unannotated cell types'
    )

    # Training parameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=10.0,
        help='Ridge regularization parameter (default: 10.0)'
    )
    parser.add_argument(
        '--tune_alpha',
        action='store_true',
        help='Tune alpha using cross-validation'
    )
    parser.add_argument(
        '--alpha_values',
        type=float,
        nargs='+',
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
        help='Alpha values to try when tuning (default: 0.1 1 10 100 1000)'
    )
    parser.add_argument(
        '--use_log_transform',
        action='store_true',
        default=True,
        help='Log-transform expression values (default: True)'
    )
    parser.add_argument(
        '--pseudocount',
        type=float,
        default=1.0,
        help='Pseudocount for log transformation (default: 1.0)'
    )
    parser.add_argument(
        '--use_weights',
        action='store_true',
        help='Weight genes by expression variance'
    )

    # Data splitting
    parser.add_argument(
        '--test_chroms',
        type=str,
        nargs='+',
        default=['8', '9'],
        help='Chromosomes for test set (default: 8 9)'
    )
    parser.add_argument(
        '--val_fraction',
        type=float,
        default=0.1,
        help='Fraction of data for validation (default: 0.1)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=3,
        help='Random seed (default: 3)'
    )

    # Cross-validation
    parser.add_argument(
        '--run_cv',
        action='store_true',
        help='Run cross-validation'
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run main pipeline
    main(args)
