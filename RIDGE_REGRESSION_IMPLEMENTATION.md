# Ridge Regression for Single-Cell Expression Prediction - Implementation Summary

## Overview

This implementation adds the capability to train ridge regression models on single-cell expression data using epigenetic predictions from gReLU models as features. This enables prediction of cell-type-specific gene expression and variant effects on expression.

## Architecture

### Two-Stage Prediction Pipeline

1. **Stage 1 - gReLU Model**: DNA sequence → Epigenetic features (20,020-dimensional vectors)
2. **Stage 2 - Ridge Regression**: Epigenetic features → Cell-type-specific gene expression

### Mathematical Formulation

For each cell type *j*:

```
log(g_i,j + pseudocount) = X^T β_j + ε_i
Loss_j = Σ_i w_i * log(L(g_i,j, X^T β_j))
β_j = argmin(Loss_j + α * ||β_j||²)
```

Where:
- `g_i,j`: Expression of gene *i* in cell type *j*
- `X`: Epigenetic features from gReLU
- `β_j`: Ridge coefficients for cell type *j*
- `w_i`: Optional gene weights based on variance
- `α`: Ridge regularization parameter

## Files Added

### Core Modules

#### 1. `/src/grelu/model/epigenetic_feature_extractor.py`
**Purpose**: Extract epigenetic predictions from trained gReLU models

**Key Classes**:
- `EpigeneticFeatureExtractor`: Main class for feature extraction
  - Extract features from DNA sequences
  - Extract features for genes from genome
  - Support for intermediate layer extraction
  - Multiple aggregation methods (mean, max, sum)
  - Save/load features in pickle or HDF5 format

**Key Functions**:
- `extract_features_for_variants()`: Extract features for both reference and variant alleles

#### 2. `/src/grelu/data/single_cell.py`
**Purpose**: Handle single-cell expression data

**Key Classes**:
- `SingleCellExpressionDataset`: Dataset class managing train/val/test splits

**Key Functions**:
- `load_expression_matrix()`: Load expression data from various formats
- `aggregate_expression_by_cell_type()`: Aggregate single cells to cell types
- `prepare_data_matrices()`: Align epigenetic features with expression data
- `split_genes_by_chromosome()`: Split genes by chromosome for train/val/test
- `get_gene_weights()`: Compute gene weights based on expression variance
- `filter_cell_types()`: Filter immune and ambiguous cell types
- `check_gene_existence()`: Filter genes present in both features and expression

#### 3. `/src/grelu/model/ridge.py`
**Purpose**: Ridge regression models for expression prediction

**Key Classes**:
- `RidgeExpressionModel`: Single cell type ridge model
  - Wraps sklearn.linear_model.Ridge
  - Handles fitting, prediction, save/load

- `MultiGroupRidgeModel`: Collection of models for multiple cell types
  - Train one model per cell type
  - Predict across all cell types
  - Save/load entire model collection

**Key Functions**:
- `evaluate_predictions()`: Compute evaluation metrics (Spearman, Pearson, RMSE, R², etc.)
- `tune_alpha()`: Hyperparameter tuning with cross-validation
- `cross_validate_model()`: K-fold cross-validation
- `summarize_cv_results()`: Summarize CV results into DataFrame

#### 4. `/src/grelu/model/ridge_trainer.py`
**Purpose**: High-level training functions

**Key Functions**:
- `train_expression_models()`: Main training function with full pipeline
- `tune_and_train()`: Tune alpha then train final model
- `cross_validate_expression_models()`: Evaluate with K-fold CV
- `evaluate_on_test_set()`: Evaluate on held-out test chromosomes

#### 5. `/src/grelu/model/variant_expression.py`
**Purpose**: Predict variant effects on expression

**Key Functions**:
- `predict_variant_effects()`: Predict from pre-computed variant features
- `predict_variant_effects_on_the_fly()`: Compute features and predict in one step
- `compute_delta_representations()`: Compute L2 distance between ref and alt features
- `save_variant_effects()`: Save predictions to files
- `get_top_effects_per_cell_type()`: Find top variants per cell type
- `annotate_variants_with_effects()`: Add effect predictions to variant DataFrame

### Supporting Files

#### 6. `/src/grelu/model/__init__.py`
Updated to export all ridge regression classes and functions

#### 7. `/src/grelu/data/__init__.py`
Updated to export single-cell data utilities

#### 8. `/examples/train_ridge_expression.py`
Complete command-line script demonstrating the full workflow:
- Load gReLU model
- Extract epigenetic features
- Load expression data
- Train ridge models
- Cross-validation
- Test set evaluation
- Variant effect prediction

#### 9. `/tests/test_ridge_regression.py`
Comprehensive unit tests covering:
- RidgeExpressionModel
- MultiGroupRidgeModel
- Evaluation functions
- Single-cell data utilities
- SingleCellExpressionDataset

#### 10. `/docs/ridge_regression_guide.md`
Complete user guide with:
- Quick start examples
- Detailed API usage
- Data format requirements
- Command-line interface
- Advanced topics
- Troubleshooting
- Performance tips

## Features Implemented

### Training Features
✅ Ridge regression with customizable regularization (α)
✅ Multi-group training (one model per cell type)
✅ Weighted training based on expression variance
✅ Log2 transformation with pseudocount
✅ Hyperparameter tuning with grid search CV
✅ K-fold cross-validation
✅ Chromosome-based train/val/test splitting
✅ Model persistence (save/load)

### Evaluation Features
✅ Spearman correlation
✅ Pearson correlation
✅ RMSE
✅ R² score
✅ Explained variance
✅ Per-cell-type and aggregate metrics

### Data Handling Features
✅ Load expression from CSV, TSV, pickle
✅ Aggregate single cells by cell type
✅ Filter immune and ambiguous cell types
✅ Gene weighting by variance
✅ Automatic gene alignment between features and expression
✅ Chromosome-based data splitting

### Feature Extraction Features
✅ Extract from trained gReLU models
✅ Multiple aggregation methods (mean, max, sum, custom)
✅ Intermediate layer extraction support
✅ Batch processing for efficiency
✅ Feature caching (pickle and HDF5)
✅ Reverse complement handling for minus strand genes

### Variant Analysis Features
✅ Predict effects from pre-computed features
✅ On-the-fly feature extraction and prediction
✅ Delta epigenetic representation computation
✅ Top variant identification per cell type
✅ Variant annotation with effects
✅ Both difference (alt - ref) and ratio (alt / ref) modes

## Usage Examples

### Basic Training
```python
from grelu.model import train_expression_models

model, results, preds, truth = train_expression_models(
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    alpha=10.0,
    use_weights=True,
    save_models=True,
    model_save_dir='models/'
)
```

### Hyperparameter Tuning
```python
from grelu.model import tune_and_train

model, best_alpha, results = tune_and_train(
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    alphas=[0.1, 1.0, 10.0, 100.0],
    cv_folds=5
)
```

### Variant Effect Prediction
```python
from grelu.model import predict_variant_effects_on_the_fly

ref_preds, alt_preds, effects = predict_variant_effects_on_the_fly(
    model=model,
    grelu_model=grelu_model,
    variants_df=variants_df,
    genome_fasta='hg38.fa'
)
```

### Command-Line Usage
```bash
python examples/train_ridge_expression.py \
    --grelu_model models/grelu.ckpt \
    --genes_file data/genes.csv \
    --expression_file data/expression.csv \
    --genome_fasta data/hg38.fa \
    --output_dir results/ \
    --alpha 10.0 \
    --use_weights \
    --run_cv
```

## Data Format Requirements

### Genes File (CSV)
```
gene_name,chrom,tss,strand
GENE1,chr1,1000000,+
GENE2,chr1,2000000,-
```

### Expression Matrix (CSV)
Rows: Cell types/groups
Columns: Genes

```
,GENE1,GENE2,GENE3
T_cells,10.5,20.3,5.2
B_cells,15.2,18.9,6.8
```

### Variants File (CSV)
```
gene_name,chrom,pos,ref,alt,tss
GENE1,chr1,1000500,A,G,1000000
```

## Testing

Run unit tests:
```bash
pytest tests/test_ridge_regression.py -v
```

## Integration with Existing gReLU

The implementation integrates seamlessly with existing gReLU functionality:

1. **Uses existing gReLU models** for feature extraction
2. **Compatible with PyTorch and PyTorch Lightning** models
3. **Follows gReLU naming conventions** and code style
4. **Uses existing I/O utilities** (e.g., `read_fasta`)
5. **Maintains separate namespaces** to avoid conflicts

## Performance Considerations

1. **Feature Caching**: Features are saved to avoid recomputation
2. **Batch Processing**: Efficient batch inference for feature extraction
3. **Memory Management**: Support for large datasets via chunking
4. **GPU Support**: Automatic GPU utilization for feature extraction
5. **Parallel Processing**: Can process multiple cell types in parallel

## Future Extensions

Potential areas for enhancement:

1. **Ensemble Models**: Support for multiple ridge models per cell type
2. **Non-linear Models**: Option to use other sklearn models (Lasso, ElasticNet)
3. **Multi-task Learning**: Joint training across cell types
4. **Active Learning**: Iterative sample selection for efficient training
5. **Uncertainty Quantification**: Prediction intervals and confidence estimates
6. **Cell Type Hierarchy**: Leverage hierarchical relationships between cell types

## Dependencies

New dependencies added:
- `scikit-learn`: Ridge regression and metrics
- `scipy`: Statistical functions (Spearman, Pearson)
- `h5py`: HDF5 file handling for variant features

Existing gReLU dependencies used:
- `torch`: Model inference
- `pandas`: Data handling
- `numpy`: Numerical operations

## Compatibility

- **Python**: 3.8+
- **PyTorch**: Compatible with existing gReLU requirements
- **sklearn**: 0.24+
- **scipy**: 1.5+

## Summary

This implementation provides a complete, production-ready solution for:
1. Training ridge regression models on single-cell expression data
2. Predicting cell-type-specific gene expression from epigenetic features
3. Analyzing variant effects on expression across cell types

The code is well-documented, tested, and integrated with the existing gReLU framework while maintaining modularity and extensibility.
