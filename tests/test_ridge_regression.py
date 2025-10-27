"""
Unit tests for ridge regression expression prediction modules.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from grelu.model.ridge import (
    RidgeExpressionModel,
    MultiGroupRidgeModel,
    evaluate_predictions,
    tune_alpha,
    cross_validate_model,
    summarize_cv_results
)
from grelu.data.single_cell import (
    prepare_data_matrices,
    split_genes_by_chromosome,
    get_gene_weights,
    check_gene_existence,
    SingleCellExpressionDataset,
    filter_cell_types
)


class TestRidgeExpressionModel:
    """Test RidgeExpressionModel class."""

    def test_init(self):
        """Test model initialization."""
        model = RidgeExpressionModel(alpha=1.0)
        assert model.alpha == 1.0
        assert model.is_fitted == False

    def test_fit_predict(self):
        """Test fitting and prediction."""
        # Create dummy data
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 100 genes, 50 features
        y = np.random.randn(100)  # Expression values

        # Fit model
        model = RidgeExpressionModel(alpha=1.0)
        model.fit(X, y, group_name='test_group')

        assert model.is_fitted == True
        assert model.group_name == 'test_group'

        # Predict
        predictions = model.predict(X)
        assert predictions.shape == (100,)

    def test_fit_with_weights(self):
        """Test weighted fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        weights = np.random.rand(100)

        model = RidgeExpressionModel(alpha=1.0)
        model.fit(X, y, sample_weight=weights)

        predictions = model.predict(X)
        assert predictions.shape == (100,)

    def test_save_load(self):
        """Test saving and loading."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Train and save
        model = RidgeExpressionModel(alpha=2.0)
        model.fit(X, y, group_name='save_test')

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model.pkl'
            model.save(str(save_path))

            # Load
            loaded_model = RidgeExpressionModel.load(str(save_path))

            assert loaded_model.alpha == 2.0
            assert loaded_model.group_name == 'save_test'
            assert loaded_model.is_fitted == True

            # Check predictions match
            pred_original = model.predict(X)
            pred_loaded = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)


class TestMultiGroupRidgeModel:
    """Test MultiGroupRidgeModel class."""

    def test_init(self):
        """Test initialization."""
        model = MultiGroupRidgeModel(alpha=1.0, group_names=['A', 'B', 'C'])
        assert model.alpha == 1.0
        assert len(model.group_names) == 3

    def test_fit_predict(self):
        """Test fitting and prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 100 genes, 50 features
        Y = np.random.randn(100, 3)  # 3 groups
        group_names = ['Group_A', 'Group_B', 'Group_C']

        # Fit
        model = MultiGroupRidgeModel(alpha=1.0, group_names=group_names)
        model.fit(X, Y, use_log_transform=False, verbose=False)

        # Check all models are trained
        assert len(model.models) == 3
        for group_name in group_names:
            assert group_name in model.models
            assert model.models[group_name].is_fitted

        # Predict as array
        predictions = model.predict(X, return_dict=False)
        assert predictions.shape == (100, 3)

        # Predict as dict
        predictions_dict = model.predict(X, return_dict=True)
        assert len(predictions_dict) == 3
        assert all(group in predictions_dict for group in group_names)

    def test_save_load(self):
        """Test saving and loading."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 2)
        group_names = ['T_cells', 'B_cells']

        # Train and save
        model = MultiGroupRidgeModel(alpha=5.0, group_names=group_names)
        model.fit(X, Y, use_log_transform=False, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Load
            loaded_model = MultiGroupRidgeModel.load(tmpdir)

            assert loaded_model.alpha == 5.0
            assert loaded_model.group_names == group_names

            # Check predictions match
            pred_original = model.predict(X)
            pred_loaded = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)


class TestEvaluationFunctions:
    """Test evaluation functions."""

    def test_evaluate_predictions(self):
        """Test prediction evaluation."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1  # Add small noise

        results = evaluate_predictions(y_true, y_pred)

        # Check all metrics are computed
        assert 'spearman' in results
        assert 'pearson' in results
        assert 'rmse' in results
        assert 'r2' in results
        assert 'explained_variance' in results

        # Check correlations are high (since we added little noise)
        assert results['spearman'] > 0.9
        assert results['pearson'] > 0.9
        assert results['r2'] > 0.8

    def test_tune_alpha(self):
        """Test alpha tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        best_alpha, cv_results = tune_alpha(
            X, y,
            alphas=[0.1, 1.0, 10.0],
            cv=3,
            verbose=False
        )

        assert best_alpha in [0.1, 1.0, 10.0]
        assert 'best_score' in cv_results
        assert 'all_scores' in cv_results


class TestSingleCellDataUtilities:
    """Test single-cell data utilities."""

    def test_prepare_data_matrices(self):
        """Test data matrix preparation."""
        # Create mock data
        genes_df = pd.DataFrame({
            'gene_name': ['GENE1', 'GENE2', 'GENE3']
        })

        epigenetic_features = {
            'GENE1': np.array([1.0, 2.0, 3.0]),
            'GENE2': np.array([4.0, 5.0, 6.0]),
            'GENE3': np.array([7.0, 8.0, 9.0])
        }

        expression_df = pd.DataFrame({
            'GENE1': [10.0, 20.0],
            'GENE2': [30.0, 40.0],
            'GENE3': [50.0, 60.0]
        }, index=['CellType1', 'CellType2'])

        dataX, dataY, group_names = prepare_data_matrices(
            genes_df, epigenetic_features, expression_df
        )

        assert dataX.shape == (3, 3)  # 3 genes, 3 features
        assert dataY.shape == (3, 2)  # 3 genes, 2 cell types
        assert len(group_names) == 2
        np.testing.assert_array_equal(group_names, ['CellType1', 'CellType2'])

    def test_split_genes_by_chromosome(self):
        """Test chromosome-based splitting."""
        genes_df = pd.DataFrame({
            'gene_name': [f'GENE{i}' for i in range(100)],
            'chrom': ['1'] * 30 + ['2'] * 30 + ['8'] * 20 + ['9'] * 20
        })

        train, val, test = split_genes_by_chromosome(
            genes_df,
            test_chroms=['8', '9'],
            val_fraction=0.2,
            random_state=42
        )

        # Test set should have chr 8 and 9
        assert len(test) == 40
        assert all(test['chrom'].isin(['8', '9']))

        # Train + val should have remaining
        assert len(train) + len(val) == 60

    def test_get_gene_weights(self):
        """Test gene weighting."""
        # Create expression data with varying variance
        np.random.seed(42)
        expression = np.random.randn(100, 10)  # 100 genes, 10 groups

        weights = get_gene_weights(expression)

        assert weights.shape == (100,)
        assert np.all(weights > 0)  # All weights should be positive

    def test_filter_cell_types(self):
        """Test cell type filtering."""
        expression_df = pd.DataFrame({
            'GENE1': [1, 2, 3, 4, 5],
            'GENE2': [6, 7, 8, 9, 10]
        }, index=['T_cells', 'B_cells', 'Unannotated', 'Macrophage', 'Neuron'])

        # Filter ambiguous
        filtered = filter_cell_types(
            expression_df,
            remove_immune=False,
            remove_ambiguous=True
        )
        assert 'Unannotated' not in filtered.index

        # Filter immune
        filtered = filter_cell_types(
            expression_df,
            remove_immune=True,
            remove_ambiguous=False
        )
        assert 'T_cells' not in filtered.index
        assert 'B_cells' not in filtered.index
        assert 'Macrophage' not in filtered.index

    def test_check_gene_existence(self):
        """Test gene existence checking."""
        genes_df = pd.DataFrame({
            'gene_name': ['GENE1', 'GENE2', 'GENE3', 'GENE4']
        })

        epigenetic_features = {
            'GENE1': np.array([1, 2, 3]),
            'GENE2': np.array([4, 5, 6])
            # GENE3 and GENE4 missing
        }

        expression_df = pd.DataFrame({
            'GENE1': [1, 2],
            'GENE3': [3, 4]
            # GENE2 and GENE4 missing
        })

        filtered = check_gene_existence(
            genes_df, epigenetic_features, expression_df
        )

        # Only GENE1 should remain (present in both)
        assert len(filtered) == 1
        assert filtered['gene_name'].iloc[0] == 'GENE1'


class TestSingleCellExpressionDataset:
    """Test SingleCellExpressionDataset class."""

    def test_dataset_creation(self):
        """Test dataset creation and splitting."""
        # Create mock data
        genes_df = pd.DataFrame({
            'gene_name': [f'GENE{i}' for i in range(100)],
            'chrom': ['1'] * 30 + ['2'] * 30 + ['8'] * 20 + ['9'] * 20
        })

        epigenetic_features = {
            f'GENE{i}': np.random.randn(50) for i in range(100)
        }

        expression_df = pd.DataFrame(
            np.random.randn(3, 100),  # 3 cell types, 100 genes
            columns=[f'GENE{i}' for i in range(100)],
            index=['CellType1', 'CellType2', 'CellType3']
        )

        # Create dataset
        dataset = SingleCellExpressionDataset(
            genes_df=genes_df,
            epigenetic_features=epigenetic_features,
            expression_df=expression_df,
            test_chroms=['8', '9'],
            val_fraction=0.2,
            random_state=42
        )

        # Check splits
        assert len(dataset.train_genes) > 0
        assert len(dataset.val_genes) > 0
        assert len(dataset.test_genes) == 40  # Chr 8 and 9

        # Check data shapes
        X_train, Y_train = dataset.get_train_data()
        assert X_train.shape[1] == 50  # Feature dimension
        assert Y_train.shape[1] == 3  # Number of cell types


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
