# Data Augmentation to Prevent MIA in TSF
Embedding-Space Data Augmentation to Prevent Membership Inference Attacks in Clinical Time Series Forecasting

This repository implements privacy-preserving techniques for clinical time series forecasting using transformer models. The approach uses embedding-space data augmentation (ZOO, ZOO-PCA, MixUp) and DP-SGD to mitigate membership inference attacks while maintaining predictive performance on MIMIC-III and eICU datasets.

## Workflow Overview

The codebase follows a 4-stage pipeline:
1. **Data Preprocessing**: Extract and process clinical time series data
2. **Model Training**: Train transformer models for forecasting
3. **Privacy Enhancement**: Apply privacy-preserving data augmentation techniques  
4. **Results Evaluation**: Assess privacy-utility tradeoffs and generate results

## 1. Data Preprocessing
- `01_preprocess_mimic.py`
- `01_preprocess_eicu.py`

The MIMIC preprocessing is based on Tipirneni & Reddy, 2021. Both extract from their respective datasets all patient stays where a patient is on ICU for more than 48 hours. 131 (MIMIC) and 100 (eICU) variables are exported to `<name>_preprocessed.pkl`, then binned such that there is no more than one value per hour `<name>_binned_processed.pkl`, and finally prepared as `fore_op_<train/valid/test>.pkl` for model training and evaluation.

## 2. Model Training
- `02_train_mimic.py`
- `02_train_eicu.py`


## 3. Privacy Enhancement
- `03_privacy_mimic.py`
- `03_privacy_eicu.py`
- `04_create_privacy_mimic.py`
- `04_create_privacy_eicu.py`

Similar to the training stage, `03_privacy_<name>.py` contains the privacy fine-tuning implementation and `04_create_privacy_<name>.py` manages configurations. Available algorithms include 'zoo', 'zoo-pca', 'mixup', 'only', and 'dp-sgd' with configurable hyperparameters.

## 4. Results Evaluation
- `05_loss_lists.py`

The `05_loss_lists.py` calculates the losses for train, validation, and test sets.

## Core Model Components

### Model Architecture (`models/`)
- **`InformerAutoregressive.py`**: Core Informer model with autoregressive decoder for time series forecasting, supporting both training and inference modes with configurable embedding types and attention mechanisms.

- **`InformerAutoregressive_emb.py`**: Modified Informer variant that accepts pre-computed embeddings as input, designed for embedding-space data augmentation experiments.

### Neural Network Layers (`layers/`)
- **`Embed.py`**: Comprehensive embedding modules including positional, temporal, and token embeddings with various combinations.

- **`SelfAttention_Family.py`**: Attention mechanism implementations:
  - `FullAttention`: Standard multi-head attention with optional causal masking
  - `ProbAttention`: Sparse attention with probabilistic sampling for efficiency
  - `AttentionLayer`: Wrapper for attention mechanisms with projection layers

- **`Transformer_EncDec.py`**: Transformer encoder-decoder building blocks with attention and feed-forward components, convolutional layers for sequence length reduction, and support for distillation.

### Support Utilities (`utils/`)
- **`config.py`**: Argument parser for transformer-based time series forecasting models, supporting configurable parameters for model dimensions, layers, training settings, and data processing options.

- **`masking.py`**: Attention masking utilities including `TriangularCausalMask` for autoregressive modeling and `ProbMask` for probabilistic sparse attention mechanisms.

- **`pca.py`**: Principal Component Analysis utilities for embedding space analysis and perturbation:
  - Variance explanation analysis in time series embeddings
  - Batched PCA implementation across time dimensions
  - Noise injection along principal component directions for ZOO-PCA data augmentation
  - Support for both scikit-learn and custom SVD-based PCA implementations
 

## Citation
When using this paper or code, please use:

```
@article{fracarolliETAL25MIA,
  title={Embedding-Space Data Augmentation to Prevent Membership Inference Attacks in Clinical Time Series Forecasting},
  author={Fracarolli, Marius and Staniek, Michael and Riezler, Stefan},
  journal={Proceedings of Machine Learning Research},
  volume={287},
  pages={1--14},
  year={2025}
}
```
