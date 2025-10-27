"""
Ridge regression models for predicting gene expression from epigenetic features.

This module implements ridge regression to map epigenetic predictions from
gReLU models to cell-type-specific gene expression predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import pickle
import re
from collections import defaultdict
from sklearn import linear_model
from sklearn.model_selection import KFold, GridSearchCV
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from tqdm import tqdm
import warnings


class RidgeExpressionModel:
    """
    Ridge regression model for predicting gene expression from epigenetic features.

    This model maps epigenetic features (from gReLU) to gene expression for a
    single cell type or experimental group.

    Args:
        alpha: Regularization strength (default: 1.0)
        fit_intercept: Whether to fit intercept (default: True)
        max_iter: Maximum iterations for solver (default: None)
        solver: Solver to use (default: 'auto')

    Attributes:
        model: Trained sklearn Ridge model
        group_name: Name of the cell type/group this model predicts
        is_fitted: Whether the model has been fitted

    Example:
        >>> model = RidgeExpressionModel(alpha=1.0)
        >>> model.fit(X_train, y_train, group_name='T_cells')
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: Optional[int] = None,
        solver: str = 'auto'
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.solver = solver
        self.model = None
        self.group_name = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        group_name: Optional[str] = None
    ):
        """
        Fit the ridge regression model.

        Args:
            X: (n_samples, n_features) array of epigenetic features
            y: (n_samples,) array of expression values
            sample_weight: Optional (n_samples,) array of sample weights
            group_name: Name of the group/cell type
        """
        self.model = linear_model.Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            solver=self.solver
        )

        self.model.fit(X, y, sample_weight=sample_weight)
        self.group_name = group_name
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict gene expression from epigenetic features.

        Args:
            X: (n_samples, n_features) array of epigenetic features

        Returns:
            (n_samples,) array of predicted expression values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def save(self, output_path: str):
        """Save model to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'alpha': self.alpha,
                'group_name': self.group_name,
                'is_fitted': self.is_fitted
            }, f)

    @classmethod
    def load(cls, input_path: str):
        """Load model from file."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(alpha=data['alpha'])
        instance.model = data['model']
        instance.group_name = data.get('group_name')
        instance.is_fitted = data.get('is_fitted', True)
        return instance


class MultiGroupRidgeModel:
    """
    Collection of ridge regression models for multiple cell types/groups.

    This class manages training and prediction for multiple cell types,
    with one ridge model per cell type.

    Args:
        alpha: Regularization strength for all models
        group_names: Names of cell types/groups to train models for

    Attributes:
        models: Dictionary mapping group names to RidgeExpressionModel instances
        group_names: List of group/cell type names

    Example:
        >>> multi_model = MultiGroupRidgeModel(alpha=1.0, group_names=['T_cells', 'B_cells'])
        >>> multi_model.fit(X_train, Y_train)
        >>> predictions = multi_model.predict(X_test)  # Returns dict of predictions
    """

    def __init__(
        self,
        alpha: float = 1.0,
        group_names: Optional[List[str]] = None
    ):
        self.alpha = alpha
        self.group_names = group_names or []
        self.models = {}

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        group_names: Optional[List[str]] = None,
        sample_weights: Optional[np.ndarray] = None,
        use_log_transform: bool = True,
        pseudocount: float = 1.0,
        verbose: bool = True
    ):
        """
        Fit ridge models for all groups.

        Args:
            X: (n_genes, n_features) array of epigenetic features
            Y: (n_genes, n_groups) array of expression values
            group_names: Names of groups (if not provided in __init__)
            sample_weights: Optional (n_genes,) array of gene weights
            use_log_transform: Whether to log-transform expression (recommended)
            pseudocount: Pseudocount for log transformation
            verbose: Show progress
        """
        if group_names is not None:
            self.group_names = group_names
        elif not self.group_names:
            raise ValueError("group_names must be provided")

        if len(self.group_names) != Y.shape[1]:
            raise ValueError(f"Number of groups ({len(self.group_names)}) "
                           f"does not match Y shape ({Y.shape[1]})")

        iterator = enumerate(self.group_names)
        if verbose:
            iterator = tqdm(list(iterator), desc="Training models")

        for i, group_name in iterator:
            # Get expression for this group
            y = Y[:, i]

            # Log transform if requested
            if use_log_transform:
                y = np.log2(y + pseudocount)

            # Create and fit model
            model = RidgeExpressionModel(alpha=self.alpha)
            model.fit(X, y, sample_weight=sample_weights, group_name=group_name)

            self.models[group_name] = model

    def predict(
        self,
        X: np.ndarray,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict expression for all groups.

        Args:
            X: (n_genes, n_features) array of epigenetic features
            return_dict: If True, return dict mapping group names to predictions.
                        If False, return (n_genes, n_groups) array

        Returns:
            Predictions for all groups
        """
        predictions = {}

        for group_name, model in self.models.items():
            predictions[group_name] = model.predict(X)

        if return_dict:
            return predictions
        else:
            # Convert to array
            pred_array = np.zeros((X.shape[0], len(self.group_names)))
            for i, group_name in enumerate(self.group_names):
                pred_array[:, i] = predictions[group_name]
            return pred_array

    def save(self, output_dir: str):
        """
        Save all models to directory.

        Each model is saved as a separate file named <group_name>.pkl
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for group_name, model in self.models.items():
            # Sanitize filename
            filename = re.sub(r'[^\w]', '_', group_name) + '.pkl'
            model.save(str(output_path / filename))

        # Save metadata
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'group_names': self.group_names,
                'alpha': self.alpha
            }, f)

    @classmethod
    def load(cls, input_dir: str):
        """Load all models from directory."""
        input_path = Path(input_dir)

        # Load metadata
        with open(input_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(
            alpha=metadata['alpha'],
            group_names=metadata['group_names']
        )

        # Load models
        for group_name in metadata['group_names']:
            filename = re.sub(r'[^\w]', '_', group_name) + '.pkl'
            model_path = input_path / filename
            if model_path.exists():
                instance.models[group_name] = RidgeExpressionModel.load(str(model_path))

        return instance


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = ['spearman', 'pearson', 'rmse', 'r2', 'explained_variance']
) -> Dict[str, float]:
    """
    Evaluate prediction performance.

    Args:
        y_true: True expression values
        y_pred: Predicted expression values
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric names to values
    """
    results = {}

    if 'spearman' in metrics:
        corr, _ = spearmanr(y_true, y_pred)
        results['spearman'] = corr

    if 'pearson' in metrics:
        corr, _ = pearsonr(y_true, y_pred)
        results['pearson'] = corr

    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)

    if 'explained_variance' in metrics:
        results['explained_variance'] = explained_variance_score(y_true, y_pred)

    return results


def tune_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: Optional[List[float]] = None,
    cv: int = 5,
    sample_weights: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Tune alpha hyperparameter using cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        alphas: List of alpha values to try. If None, uses default range.
        cv: Number of cross-validation folds
        sample_weights: Optional sample weights
        verbose: Print progress

    Returns:
        Tuple of (best_alpha, cv_results)
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    ridge = linear_model.Ridge()
    param_grid = {'alpha': alphas}

    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1 if verbose else 0
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    best_alpha = grid_search.best_params_['alpha']
    cv_results = {
        'best_alpha': best_alpha,
        'best_score': grid_search.best_score_,
        'all_scores': grid_search.cv_results_
    }

    return best_alpha, cv_results


def cross_validate_model(
    X: np.ndarray,
    Y: np.ndarray,
    group_names: List[str],
    n_folds: int = 5,
    alpha: float = 1.0,
    use_log_transform: bool = True,
    pseudocount: float = 1.0,
    sample_weights: Optional[np.ndarray] = None,
    random_state: int = 3,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Perform K-fold cross-validation.

    Args:
        X: (n_genes, n_features) array of epigenetic features
        Y: (n_genes, n_groups) array of expression values
        group_names: List of group/cell type names
        n_folds: Number of CV folds
        alpha: Ridge regularization parameter
        use_log_transform: Whether to log-transform expression
        pseudocount: Pseudocount for log transform
        sample_weights: Optional gene weights
        random_state: Random seed
        verbose: Show progress

    Returns:
        Tuple of (results_dict, predictions_dict) where:
            results_dict: {fold: {group: metrics}}
            predictions_dict: {fold: {'truth': array, 'preds': array, 'groups': list}}
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {}
    predictions = {}

    fold_iterator = enumerate(kfold.split(X))
    if verbose:
        fold_iterator = tqdm(list(fold_iterator), desc="Cross-validation")

    for fold, (train_idx, val_idx) in fold_iterator:
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        weights_train = sample_weights[train_idx] if sample_weights is not None else None

        # Train multi-group model
        model = MultiGroupRidgeModel(alpha=alpha, group_names=group_names)
        model.fit(
            X_train, Y_train,
            sample_weights=weights_train,
            use_log_transform=use_log_transform,
            pseudocount=pseudocount,
            verbose=False
        )

        # Predict
        Y_pred = model.predict(X_val)

        # Log transform validation data if needed
        if use_log_transform:
            Y_val_transformed = np.log2(Y_val + pseudocount)
        else:
            Y_val_transformed = Y_val

        # Evaluate each group
        results[fold] = {}
        for i, group_name in enumerate(group_names):
            y_true = Y_val_transformed[:, i]
            y_pred = Y_pred[:, i]

            group_results = evaluate_predictions(y_true, y_pred)
            results[fold][group_name] = group_results

        # Store predictions
        predictions[fold] = {
            'truth': Y_val_transformed,
            'preds': Y_pred,
            'groups': group_names,
            'genes_idx': val_idx
        }

    return results, predictions


def summarize_cv_results(cv_results: Dict) -> pd.DataFrame:
    """
    Summarize cross-validation results into a DataFrame.

    Args:
        cv_results: Results from cross_validate_model

    Returns:
        DataFrame with mean and std of metrics across folds for each group
    """
    summary_data = defaultdict(list)

    # Get all groups and metrics
    first_fold = list(cv_results.keys())[0]
    groups = list(cv_results[first_fold].keys())
    metrics = list(cv_results[first_fold][groups[0]].keys())

    for group in groups:
        for metric in metrics:
            values = [cv_results[fold][group][metric] for fold in cv_results.keys()]
            summary_data[f'{metric}_mean'].append(np.mean(values))
            summary_data[f'{metric}_std'].append(np.std(values))

    summary_df = pd.DataFrame(summary_data, index=groups)
    return summary_df
