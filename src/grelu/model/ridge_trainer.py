"""
High-level training functions for ridge regression models.

This module provides convenient functions for training ridge models to predict
gene expression from epigenetic features, including hyperparameter tuning,
cross-validation, and model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import pickle
import logging

from grelu.data.single_cell import (
    SingleCellExpressionDataset,
    get_gene_weights,
    filter_cell_types
)
from grelu.model.ridge import (
    MultiGroupRidgeModel,
    evaluate_predictions,
    tune_alpha,
    cross_validate_model,
    summarize_cv_results
)


def train_expression_models(
    genes_df: pd.DataFrame,
    epigenetic_features: Dict[str, np.ndarray],
    expression_df: pd.DataFrame,
    alpha: float = 1.0,
    test_chroms: List[str] = ['8', '9'],
    val_fraction: float = 0.1,
    use_log_transform: bool = True,
    pseudocount: float = 1.0,
    use_weights: bool = False,
    weight_percentiles: Tuple[float, float] = (70, 10),
    filter_immune: bool = False,
    filter_ambiguous: bool = False,
    save_models: bool = False,
    model_save_dir: Optional[str] = None,
    save_predictions: bool = False,
    predictions_save_dir: Optional[str] = None,
    random_state: int = 3,
    verbose: bool = True
) -> Tuple[MultiGroupRidgeModel, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Train ridge regression models for all cell types.

    This is the main training function that handles data preparation,
    model training, and evaluation.

    Args:
        genes_df: DataFrame with columns: gene_name, chrom, tss
        epigenetic_features: Dict mapping gene names to feature arrays
        expression_df: DataFrame with genes as columns, cell types as rows
        alpha: Ridge regularization parameter
        test_chroms: Chromosomes to hold out for testing
        val_fraction: Fraction of data for validation
        use_log_transform: Log-transform expression values
        pseudocount: Pseudocount for log transformation
        use_weights: Weight genes by expression variance
        weight_percentiles: (top, bottom) percentiles for weighting
        filter_immune: Remove immune cell types
        filter_ambiguous: Remove ambiguous cell types
        save_models: Save trained models
        model_save_dir: Directory to save models
        save_predictions: Save predictions
        predictions_save_dir: Directory to save predictions
        random_state: Random seed
        verbose: Print progress

    Returns:
        Tuple of (trained_model, results_dict, predictions_df, truth_df)

    Example:
        >>> model, results, preds, truth = train_expression_models(
        ...     genes_df=genes,
        ...     epigenetic_features=features,
        ...     expression_df=expr,
        ...     alpha=10.0,
        ...     use_weights=True,
        ...     save_models=True,
        ...     model_save_dir='models/ridge/'
        ... )
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting model training")

    # Filter cell types if requested
    if filter_immune or filter_ambiguous:
        expression_df = filter_cell_types(
            expression_df,
            remove_immune=filter_immune,
            remove_ambiguous=filter_ambiguous
        )

    # Create dataset
    if verbose:
        logging.info("Preparing dataset")

    dataset = SingleCellExpressionDataset(
        genes_df=genes_df,
        epigenetic_features=epigenetic_features,
        expression_df=expression_df,
        test_chroms=test_chroms,
        val_fraction=val_fraction,
        random_state=random_state,
        check_existence=True
    )

    X_train, Y_train = dataset.get_train_data()
    X_val, Y_val = dataset.get_val_data()

    if verbose:
        logging.info(f"Training set: {X_train.shape[0]} genes, {Y_train.shape[1]} cell types")
        logging.info(f"Validation set: {X_val.shape[0]} genes")

    # Compute sample weights if requested
    sample_weights = None
    if use_weights:
        sample_weights = get_gene_weights(
            Y_train,
            percentile_cutoff_top=weight_percentiles[0],
            percentile_cutoff_bottom=weight_percentiles[1]
        )
        if verbose:
            logging.info(f"Using weighted training (mean weight: {sample_weights.mean():.2f})")

    # Add macro-average cell type
    mean_expr_train = Y_train.mean(axis=1, keepdims=True)
    Y_train = np.concatenate([Y_train, mean_expr_train], axis=1)

    if len(X_val) > 0:
        mean_expr_val = Y_val.mean(axis=1, keepdims=True)
        Y_val = np.concatenate([Y_val, mean_expr_val], axis=1)

    group_names = list(dataset.group_names) + ['macro_mean']

    # Train model
    if verbose:
        logging.info("Training ridge models")

    model = MultiGroupRidgeModel(alpha=alpha, group_names=group_names)
    model.fit(
        X_train,
        Y_train,
        sample_weights=sample_weights,
        use_log_transform=use_log_transform,
        pseudocount=pseudocount,
        verbose=verbose
    )

    # Evaluate on validation set
    results = {}
    preds_all = None
    truth_all = None

    if len(X_val) > 0:
        if verbose:
            logging.info("Evaluating on validation set")

        preds_all = model.predict(X_val)

        # Transform validation labels
        if use_log_transform:
            truth_all = np.log2(Y_val + pseudocount)
        else:
            truth_all = Y_val

        # Evaluate each group
        for i, group_name in enumerate(group_names):
            y_true = truth_all[:, i]
            y_pred = preds_all[:, i]

            group_results = evaluate_predictions(y_true, y_pred)
            results[group_name] = group_results

            if verbose:
                spearman = group_results.get('spearman', np.nan)
                logging.info(f"{group_name}: Spearman = {spearman:.4f}")

        # Create DataFrames
        preds_df = pd.DataFrame(
            preds_all,
            columns=group_names,
            index=dataset.val_genes
        )

        truth_df = pd.DataFrame(
            truth_all,
            columns=group_names,
            index=dataset.val_genes
        )
    else:
        preds_df = pd.DataFrame()
        truth_df = pd.DataFrame()

    # Save models if requested
    if save_models and model_save_dir:
        if verbose:
            logging.info(f"Saving models to {model_save_dir}")
        model.save(model_save_dir)

    # Save predictions if requested
    if save_predictions and predictions_save_dir and len(preds_df) > 0:
        if verbose:
            logging.info(f"Saving predictions to {predictions_save_dir}")

        Path(predictions_save_dir).mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(Path(predictions_save_dir) / 'predictions.csv')
        truth_df.to_csv(Path(predictions_save_dir) / 'truth.csv')

        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(Path(predictions_save_dir) / 'results.csv')

    return model, results, preds_df, truth_df


def tune_and_train(
    genes_df: pd.DataFrame,
    epigenetic_features: Dict[str, np.ndarray],
    expression_df: pd.DataFrame,
    alphas: Optional[List[float]] = None,
    cv_folds: int = 5,
    test_chroms: List[str] = ['8', '9'],
    use_log_transform: bool = True,
    pseudocount: float = 1.0,
    use_weights: bool = False,
    save_models: bool = False,
    model_save_dir: Optional[str] = None,
    random_state: int = 3,
    verbose: bool = True
) -> Tuple[MultiGroupRidgeModel, float, Dict]:
    """
    Tune alpha hyperparameter and train final model.

    This function performs hyperparameter tuning using cross-validation,
    then trains a final model on all training data with the best alpha.

    Args:
        genes_df: DataFrame with gene information
        epigenetic_features: Dict of gene features
        expression_df: Expression matrix
        alphas: List of alpha values to try (default: [0.1, 1, 10, 100, 1000])
        cv_folds: Number of CV folds for tuning
        test_chroms: Chromosomes for test set
        use_log_transform: Log-transform expression
        pseudocount: Pseudocount for log
        use_weights: Use gene weighting
        save_models: Save final models
        model_save_dir: Directory for saving models
        random_state: Random seed
        verbose: Print progress

    Returns:
        Tuple of (trained_model, best_alpha, cv_results)

    Example:
        >>> model, best_alpha, cv_results = tune_and_train(
        ...     genes_df=genes,
        ...     epigenetic_features=features,
        ...     expression_df=expr,
        ...     alphas=[0.1, 1.0, 10.0, 100.0]
        ... )
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting hyperparameter tuning")

    # Prepare dataset
    dataset = SingleCellExpressionDataset(
        genes_df=genes_df,
        epigenetic_features=epigenetic_features,
        expression_df=expression_df,
        test_chroms=test_chroms,
        val_fraction=0.0,  # No validation split during tuning
        random_state=random_state,
        check_existence=True
    )

    X_train, Y_train = dataset.get_train_data()

    # Compute weights if requested
    sample_weights = None
    if use_weights:
        sample_weights = get_gene_weights(Y_train)

    # Tune on first cell type (for efficiency)
    if use_log_transform:
        y_train_first = np.log2(Y_train[:, 0] + pseudocount)
    else:
        y_train_first = Y_train[:, 0]

    if verbose:
        logger.info(f"Tuning alpha on {dataset.group_names[0]}")

    best_alpha, cv_results = tune_alpha(
        X_train,
        y_train_first,
        alphas=alphas,
        cv=cv_folds,
        sample_weights=sample_weights,
        verbose=verbose
    )

    if verbose:
        logger.info(f"Best alpha: {best_alpha}")

    # Train final model with best alpha
    model, results, _, _ = train_expression_models(
        genes_df=genes_df,
        epigenetic_features=epigenetic_features,
        expression_df=expression_df,
        alpha=best_alpha,
        test_chroms=test_chroms,
        val_fraction=0.1,
        use_log_transform=use_log_transform,
        pseudocount=pseudocount,
        use_weights=use_weights,
        save_models=save_models,
        model_save_dir=model_save_dir,
        random_state=random_state,
        verbose=verbose
    )

    return model, best_alpha, {'cv_results': cv_results, 'val_results': results}


def cross_validate_expression_models(
    genes_df: pd.DataFrame,
    epigenetic_features: Dict[str, np.ndarray],
    expression_df: pd.DataFrame,
    alpha: float = 1.0,
    n_folds: int = 5,
    test_chroms: List[str] = ['8', '9'],
    use_log_transform: bool = True,
    pseudocount: float = 1.0,
    use_weights: bool = False,
    save_predictions: bool = False,
    predictions_save_dir: Optional[str] = None,
    random_state: int = 3,
    verbose: bool = True
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Perform cross-validation to evaluate model performance.

    Args:
        genes_df: DataFrame with gene information
        epigenetic_features: Dict of gene features
        expression_df: Expression matrix
        alpha: Ridge regularization parameter
        n_folds: Number of CV folds
        test_chroms: Chromosomes to exclude (test set)
        use_log_transform: Log-transform expression
        pseudocount: Pseudocount for log
        use_weights: Use gene weighting
        save_predictions: Save CV predictions
        predictions_save_dir: Directory for saving predictions
        random_state: Random seed
        verbose: Print progress

    Returns:
        Tuple of (cv_results, predictions, summary_df)

    Example:
        >>> cv_results, predictions, summary = cross_validate_expression_models(
        ...     genes_df=genes,
        ...     epigenetic_features=features,
        ...     expression_df=expr,
        ...     n_folds=5
        ... )
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting cross-validation")

    # Prepare dataset (exclude test chromosomes)
    dataset = SingleCellExpressionDataset(
        genes_df=genes_df,
        epigenetic_features=epigenetic_features,
        expression_df=expression_df,
        test_chroms=test_chroms,
        val_fraction=0.0,  # Use all non-test data for CV
        random_state=random_state,
        check_existence=True
    )

    X_train, Y_train = dataset.get_train_data()
    group_names = list(dataset.group_names)

    # Compute weights if requested
    sample_weights = None
    if use_weights:
        sample_weights = get_gene_weights(Y_train)

    # Perform cross-validation
    cv_results, predictions = cross_validate_model(
        X=X_train,
        Y=Y_train,
        group_names=group_names,
        n_folds=n_folds,
        alpha=alpha,
        use_log_transform=use_log_transform,
        pseudocount=pseudocount,
        sample_weights=sample_weights,
        random_state=random_state,
        verbose=verbose
    )

    # Summarize results
    summary_df = summarize_cv_results(cv_results)

    if verbose:
        logger.info("\nCross-validation results:")
        logger.info(f"\n{summary_df[['spearman_mean', 'spearman_std']]}")

    # Save if requested
    if save_predictions and predictions_save_dir:
        Path(predictions_save_dir).mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_df.to_csv(Path(predictions_save_dir) / 'cv_summary.csv')

        # Save detailed results
        with open(Path(predictions_save_dir) / 'cv_results.pkl', 'wb') as f:
            pickle.dump({'results': cv_results, 'predictions': predictions}, f)

    return cv_results, predictions, summary_df


def evaluate_on_test_set(
    model: MultiGroupRidgeModel,
    genes_df: pd.DataFrame,
    epigenetic_features: Dict[str, np.ndarray],
    expression_df: pd.DataFrame,
    test_chroms: List[str] = ['8', '9'],
    use_log_transform: bool = True,
    pseudocount: float = 1.0,
    save_predictions: bool = False,
    predictions_save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate trained model on held-out test set.

    Args:
        model: Trained MultiGroupRidgeModel
        genes_df: DataFrame with gene information
        epigenetic_features: Dict of gene features
        expression_df: Expression matrix
        test_chroms: Test chromosomes
        use_log_transform: Log-transform expression
        pseudocount: Pseudocount for log
        save_predictions: Save test predictions
        predictions_save_dir: Directory for saving predictions
        verbose: Print progress

    Returns:
        Tuple of (results_dict, predictions_df, truth_df)
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Evaluating on test set")

    # Get test data
    dataset = SingleCellExpressionDataset(
        genes_df=genes_df,
        epigenetic_features=epigenetic_features,
        expression_df=expression_df,
        test_chroms=test_chroms,
        val_fraction=0.0,
        random_state=3,
        check_existence=True
    )

    X_test, Y_test = dataset.get_test_data()

    if len(X_test) == 0:
        raise ValueError("No test data found")

    if verbose:
        logger.info(f"Test set: {X_test.shape[0]} genes")

    # Predict
    preds = model.predict(X_test)

    # Transform labels
    if use_log_transform:
        truth = np.log2(Y_test + pseudocount)
    else:
        truth = Y_test

    # Evaluate
    results = {}
    for i, group_name in enumerate(model.group_names):
        if i >= truth.shape[1]:  # Skip macro_mean if not in test set
            continue

        y_true = truth[:, i]
        y_pred = preds[:, i]

        group_results = evaluate_predictions(y_true, y_pred)
        results[group_name] = group_results

        if verbose:
            spearman = group_results.get('spearman', np.nan)
            logger.info(f"{group_name}: Spearman = {spearman:.4f}")

    # Create DataFrames
    preds_df = pd.DataFrame(
        preds[:, :truth.shape[1]],
        columns=model.group_names[:truth.shape[1]],
        index=dataset.test_genes
    )

    truth_df = pd.DataFrame(
        truth,
        columns=model.group_names[:truth.shape[1]],
        index=dataset.test_genes
    )

    # Save if requested
    if save_predictions and predictions_save_dir:
        Path(predictions_save_dir).mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(Path(predictions_save_dir) / 'test_predictions.csv')
        truth_df.to_csv(Path(predictions_save_dir) / 'test_truth.csv')

        results_df = pd.DataFrame(results).T
        results_df.to_csv(Path(predictions_save_dir) / 'test_results.csv')

    return results, preds_df, truth_df
