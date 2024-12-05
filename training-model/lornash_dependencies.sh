mamba create --name lornash -y pip "python==3.10" numpy

conda activate lornash

mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install packaging ninja wheel
mamba install -y nvidia/label/cuda-12.4.0::cuda-toolkit libxcrypt

pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/rotary
# this cannot be installed with pip
# pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/layer_norm
mamba install -c nvidia -c pytorch conda-forge::flash-attn-layer-norm=2.6.3 pytorch pytorch-cuda=12.4

pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# in a device with GPU
pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
pip install git+https://github.com/HazyResearch/flash-fft-conv.git

pip install triton evo-model transformers tokenizers accelerate datasets evaluate wandb
mamba install -c conda-forge -c bioconda r-optparse r-data.table r-stringr bioconductor-rtracklayer bioconductor-bsgenome.hsapiens.ucsc.hg38