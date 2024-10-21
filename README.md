# LoRNA-SH

LoRNA-SH is a long-context RNA foundation model for predicting transcriptome architecture. This repository contains the training pipeline and model weights.

## Repository Structure

```
lorna-sh
├── processing-seqs/          # Data preprocessing scripts
│   ├── lornash_tokenizer.json    # Tokenizer configuration
│   ├── prepare_data.R            # Data preparation script
│   └── tokenize_data.py          # Sequence tokenization script
├── training-model/          # Model training scripts
│   ├── configuration_hyena.py     # Model configuration
│   ├── generate_seqs.py          # Sequence generation script
│   ├── get_embeddings.py         # Embedding extraction
│   ├── get_likelihoods.py        # Likelihood computation
│   ├── lornash_dependencies.sh    # Dependencies installation
│   ├── modeling_hyena.py         # Core model architecture
│   └── train_model.py            # Training script
└── weights/                # Model weights
    └── model.safetensors       # Trained model weights
```