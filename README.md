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

## Usage
1. installation (better manually)
```{shell}
mamba create --name lornash python=3.8
conda activate lornash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install packaging ninja wheel
mamba install nvidia/label/cuda-12.1.0::cuda-toolkit -y

N_CORES=5
MAX_JOBS=$N_CORES pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
MAX_JOBS=$N_CORES pip install git+https://github.com/HazyResearch/flash-fft-conv.git

MAX_JOBS=$N_CORES pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0.post2
MAX_JOBS=$N_CORES pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0.post2#subdirectory=csrc/rotary
MAX_JOBS=$N_CORES pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0.post2#subdirectory=csrc/layer_norm

mamba install -c conda-forge -c bioconda r-optparse r-data.table r-stringr bioconductor-rtracklayer bioconductor-bsgenome.hsapiens.ucsc.hg38 rust -y
pip install triton evo-model transformers tokenizers accelerate datasets evaluate wandb
```

```
bash training-model/lornash_dependencies.sh
```
2. Prepare pre-mRNA sequences
```
Rscript processing-seqs/prepare_data.R \
            --species="human" \
            --gtf_file="example-CD44.gtf.gz" \
            --output_file="example-CD44.preprocessed.csv.gz"
```
3. Tokenize pre-mRNA sequences
```
python processing-seqs/tokenize_data.py \
            --data="example-CD44.preprocessed.csv.gz" \
            --tokenizer="processing-seqs/lornash_tokenizer.json" \
            --num_threads=1 \
            --tokenization_output_dir="tokenization"
```
4. Get likelihoods
```
WANDB_MODE=disabled python training-model/get_likelihoods.py \
            --checkpoint="weights/model.safetensors" \
            --tokenizer="processing-seqs/lornash_tokenizer.json" \
            --dataset="tokenization/example-CD44.preprocessed" \
            --dataset_type="example" \
            --predictions_dir="predictions" \
            --num_shards=1 \
            --num_threads=4 \
```
