"""
`grelu.model` defines the model architectures used for sequence-to-function
deep learning models, as well as ridge regression models for predicting
gene expression from epigenetic features.
"""

# Ridge regression for expression prediction
from grelu.model.ridge import (
    RidgeExpressionModel,
    MultiGroupRidgeModel,
    evaluate_predictions,
    tune_alpha,
    cross_validate_model,
    summarize_cv_results
)

from grelu.model.ridge_trainer import (
    train_expression_models,
    tune_and_train,
    cross_validate_expression_models,
    evaluate_on_test_set
)

from grelu.model.epigenetic_feature_extractor import (
    EpigeneticFeatureExtractor,
    extract_features_for_variants
)

from grelu.model.variant_expression import (
    predict_variant_effects,
    predict_variant_effects_on_the_fly,
    compute_delta_representations,
    save_variant_effects,
    get_top_effects_per_cell_type,
    annotate_variants_with_effects
)

__all__ = [
    # Ridge regression models
    'RidgeExpressionModel',
    'MultiGroupRidgeModel',
    'evaluate_predictions',
    'tune_alpha',
    'cross_validate_model',
    'summarize_cv_results',
    # Training functions
    'train_expression_models',
    'tune_and_train',
    'cross_validate_expression_models',
    'evaluate_on_test_set',
    # Feature extraction
    'EpigeneticFeatureExtractor',
    'extract_features_for_variants',
    # Variant effects
    'predict_variant_effects',
    'predict_variant_effects_on_the_fly',
    'compute_delta_representations',
    'save_variant_effects',
    'get_top_effects_per_cell_type',
    'annotate_variants_with_effects',
]
