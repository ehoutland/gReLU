# Ridge Regression for Gene Expression Prediction

This guide explains how to use gReLU's ridge regression module to predict cell-type-specific gene expression from epigenetic features.

## Overview

The ridge regression module implements a two-stage prediction pipeline:

1. **Stage 1 (gReLU Model)**: DNA sequence → Epigenetic features
2. **Stage 2 (Ridge Regression)**: Epigenetic features → Gene expression

This approach allows you to:
- Predict gene expression across multiple cell types
- Predict how genetic variants affect expression
- Train interpretable linear models on top of deep learning features

## Mathematical Formulation

For each cell type *j*, we train a separate ridge regression model:

```
log(g_i,j + pseudocount) = X^T β_j + ε_i

Loss_j = Σ_i w_i * log(L(g_i,j, X^T β_j))

β_j = argmin(Loss_j + α * ||β_j||²)
```

Where:
- `g_i,j` is the expression of gene *i* in cell type *j*
- `X` are the epigenetic features from the gReLU model
- `β_j` are the ridge regression coefficients for cell type *j*
- `w_i` are optional gene weights based on expression variance
- `α` is the ridge regularization parameter

## Installation

The ridge regression module is included with gReLU. Make sure you have the required dependencies:

```bash
pip install scikit-learn scipy pandas numpy h5py tqdm
```

## Quick Start

### 1. Extract Epigenetic Features

First, extract features from a trained gReLU model:

```python
import pandas as pd
import torch
from grelu.model import EpigeneticFeatureExtractor

# Load trained gReLU model
grelu_model = torch.load('path/to/grelu_model.pt')

# Load gene list
genes_df = pd.read_csv('genes.csv')  # Must have: gene_name, chrom, tss

# Extract features
extractor = EpigeneticFeatureExtractor(
    model=grelu_model,
    genome_fasta='hg38.fa',
    aggregation='mean'
)

features = extractor.extract_features_from_genes(
    genes_df=genes_df,
    seq_len=20000,
    batch_size=32
)

# Save for later use
extractor.save_features(features, 'epigenetic_features.pkl')
```

### 2. Prepare Expression Data

Load and optionally preprocess your single-cell expression data:

```python
from grelu.data import load_expression_matrix, filter_cell_types

# Load expression matrix (genes as columns, cell types as rows)
expression_df = load_expression_matrix('expression.csv')

# Optional: Filter cell types
expression_df = filter_cell_types(
    expression_df,
    remove_immune=True,
    remove_ambiguous=True
)
```

### 3. Train Ridge Models

Train one ridge model per cell type:

```python
from grelu.model import train_expression_models

model, results, preds, truth = train_expression_models(
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    alpha=10.0,
    test_chroms=['8', '9'],
    use_log_transform=True,
    use_weights=True,
    save_models=True,
    model_save_dir='models/ridge/'
)

# Print results
for cell_type, metrics in results.items():
    print(f"{cell_type}: Spearman = {metrics['spearman']:.4f}")
```

### 4. Predict Variant Effects

Predict how genetic variants affect expression:

```python
from grelu.model import predict_variant_effects_on_the_fly

# Load variants
variants_df = pd.read_csv('variants.csv')
# Required columns: gene_name, chrom, pos, ref, alt, tss

# Predict effects
ref_preds, alt_preds, effects = predict_variant_effects_on_the_fly(
    model=model,
    grelu_model=grelu_model,
    variants_df=variants_df,
    genome_fasta='hg38.fa',
    use_diff=True  # Return alt - ref
)

# effects is a DataFrame with variants × cell types
print(effects.head())
```

## Detailed Usage

### Hyperparameter Tuning

Tune the regularization parameter α using cross-validation:

```python
from grelu.model import tune_and_train

model, best_alpha, results = tune_and_train(
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
    cv_folds=5,
    save_models=True,
    model_save_dir='models/ridge_tuned/'
)

print(f"Best alpha: {best_alpha}")
```

### Cross-Validation

Evaluate model performance with K-fold cross-validation:

```python
from grelu.model import cross_validate_expression_models

cv_results, predictions, summary = cross_validate_expression_models(
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    alpha=10.0,
    n_folds=5,
    use_weights=True
)

# summary is a DataFrame with mean and std of metrics per cell type
print(summary[['spearman_mean', 'spearman_std']])
```

### Test Set Evaluation

Evaluate on a held-out test set:

```python
from grelu.model import evaluate_on_test_set

test_results, test_preds, test_truth = evaluate_on_test_set(
    model=model,
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    test_chroms=['8', '9']
)

for cell_type, metrics in test_results.items():
    print(f"{cell_type}: Spearman = {metrics['spearman']:.4f}")
```

### Using Pre-computed Variant Features

If you have pre-computed features for variants stored in HDF5 files:

```python
from grelu.model import predict_variant_effects

ref_preds, alt_preds, effects = predict_variant_effects(
    model=model,
    variant_features_dir='condensed_features/',
    variants_df=variants_df,
    use_diff=True
)
```

### Analyzing Variant Effects

Find top variants for each cell type:

```python
from grelu.model import get_top_effects_per_cell_type

top_variants = get_top_effects_per_cell_type(
    effects_df=effects,
    n_top=100,
    use_abs=True
)

for cell_type, variants in top_variants.items():
    print(f"\nTop 10 variants for {cell_type}:")
    print(variants.head(10))
```

Annotate variants with effects:

```python
from grelu.model import annotate_variants_with_effects

annotated = annotate_variants_with_effects(
    variants_df=variants_df,
    effects_df=effects,
    effect_threshold=0.5,  # Optional filtering
    max_effect_column=True
)

# Now has columns for each cell type + max_effect_cell_type
print(annotated[['gene_name', 'max_effect_cell_type', 'max_effect_value']])
```

## Data Format Requirements

### Genes DataFrame

Must contain:
- `gene_name`: Gene identifier
- `chrom`: Chromosome (with or without 'chr' prefix)
- `tss`: Transcription start site position

Optional:
- `strand`: '+' or '-' (for reverse complement)

Example:
```csv
gene_name,chrom,tss,strand
GENE1,chr1,1000000,+
GENE2,chr1,2000000,-
```

### Expression Matrix

- **Rows**: Cell types or individual cells
- **Columns**: Genes (matching gene names in genes_df)
- **Values**: Expression counts (will be log-transformed)

Can be provided as CSV, TSV, or pickle. Use `transpose=True` if genes are rows.

Example (genes as columns):
```csv
,GENE1,GENE2,GENE3
T_cells,10.5,20.3,5.2
B_cells,15.2,18.9,6.8
```

### Variants DataFrame

Required columns:
- `gene_name`: Gene identifier
- `chrom`: Chromosome
- `pos`: Variant position
- `ref`: Reference allele
- `alt`: Alternate allele
- `tss`: Gene TSS position

Optional:
- `SNP`: Variant identifier (will be created from chrom:pos if missing)

## Command-Line Interface

Use the provided example script:

```bash
python examples/train_ridge_expression.py \
    --grelu_model models/grelu_model.ckpt \
    --genes_file data/genes.csv \
    --expression_file data/expression.csv \
    --genome_fasta data/hg38.fa \
    --output_dir results/ridge/ \
    --alpha 10.0 \
    --use_weights \
    --run_cv \
    --variants_file data/variants.csv
```

See `python examples/train_ridge_expression.py --help` for all options.

## Advanced Topics

### Custom Feature Extraction

Extract features from an intermediate layer:

```python
extractor = EpigeneticFeatureExtractor(
    model=grelu_model,
    genome_fasta='hg38.fa',
    layer_name='model.trunk.5',  # Extract from specific layer
    aggregation='max'  # Use max pooling instead of mean
)
```

### Custom Aggregation

Provide a custom aggregation function:

```python
def custom_agg(features):
    # features shape: (batch, tasks, length)
    return features[:, :, 1000:2000].mean(dim=-1)  # Mean over positions 1000-2000

extractor = EpigeneticFeatureExtractor(
    model=grelu_model,
    genome_fasta='hg38.fa',
    aggregation=custom_agg
)
```

### Gene Weighting

Control how genes are weighted by expression variance:

```python
from grelu.data import get_gene_weights

# Custom percentiles for weighting
weights = get_gene_weights(
    expression_array,
    percentile_cutoff_top=80,    # Top 20% genes get +1 weight
    percentile_cutoff_bottom=20  # Bottom 20% genes get -0.5 weight
)
```

### Working with SingleCellExpressionDataset

For more control over data handling:

```python
from grelu.data import SingleCellExpressionDataset

dataset = SingleCellExpressionDataset(
    genes_df=genes_df,
    epigenetic_features=features,
    expression_df=expression_df,
    test_chroms=['8', '9'],
    val_fraction=0.15,
    random_state=42
)

# Access data splits
X_train, Y_train = dataset.get_train_data()
X_val, Y_val = dataset.get_val_data()
X_test, Y_test = dataset.get_test_data()

# Access gene lists
print(f"Training genes: {dataset.train_genes}")
print(f"Cell types: {dataset.group_names}")

# Save dataset
dataset.save('dataset.pkl')
```

## Performance Tips

1. **Feature Caching**: Always save extracted features to avoid recomputing:
   ```python
   extractor.save_features(features, 'features.pkl')
   features = EpigeneticFeatureExtractor.load_features('features.pkl')
   ```

2. **Batch Size**: Increase batch size for faster feature extraction if you have GPU memory:
   ```python
   features = extractor.extract_features_from_genes(
       genes_df=genes_df,
       batch_size=128  # Increase if possible
   )
   ```

3. **Gene Filtering**: Remove genes with very low expression before training:
   ```python
   # Keep genes with mean expression > threshold
   expressed_genes = expression_df.columns[expression_df.mean(axis=0) > 1.0]
   expression_df = expression_df[expressed_genes]
   ```

4. **Parallel Prediction**: For large variant sets, process in batches:
   ```python
   batch_size = 1000
   all_effects = []

   for i in range(0, len(variants_df), batch_size):
       batch = variants_df.iloc[i:i+batch_size]
       _, _, effects = predict_variant_effects_on_the_fly(
           model=model,
           grelu_model=grelu_model,
           variants_df=batch,
           genome_fasta=genome_fasta
       )
       all_effects.append(effects)

   effects_df = pd.concat(all_effects)
   ```

## Troubleshooting

### Issue: "Gene not found in epigenetic_features"

**Solution**: Ensure gene names match exactly between genes_df and expression_df. Check for:
- Capitalization differences
- Extra whitespace
- Different gene ID systems (e.g., Ensembl vs. gene symbols)

### Issue: Low correlation scores

**Potential causes**:
1. Alpha too high or too low → Try tuning with `tune_and_train()`
2. Expression data not properly normalized → Check data distribution
3. Feature extraction settings not optimal → Try different aggregation methods
4. Not enough training data → Check train/val/test split sizes

### Issue: Out of memory during feature extraction

**Solutions**:
1. Reduce batch size: `batch_size=16` or `batch_size=8`
2. Use CPU instead: `device='cpu'`
3. Process genes in chunks and save intermediate features

### Issue: Variant predictions all zero or NaN

**Causes**:
1. Variant features not found → Check HDF5 files or sequence extraction
2. Position mismatch → Verify TSS positions match between training and variants
3. Model not fitted → Ensure you called `model.fit()` before predicting

## Citation

If you use this ridge regression module, please cite:

```bibtex
@article{grelu2024,
  title={gReLU: Gene Regulatory Element Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## API Reference

See the full API documentation for detailed function signatures and parameters:

- `grelu.model.ridge`: Ridge regression models
- `grelu.model.ridge_trainer`: Training functions
- `grelu.model.epigenetic_feature_extractor`: Feature extraction
- `grelu.model.variant_expression`: Variant effect prediction
- `grelu.data.single_cell`: Single-cell data utilities
