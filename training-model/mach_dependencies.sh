conda create -p /envs/mach-1 -y
conda activate /envs/mach-1

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# pip install flash-attn==2.5.8 --no-build-isolation
pip install packaging
pip install ninja

mkdir /envs/mach-1/sources

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.8

# Set CUDA architectures for your H100
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export MAX_JOBS=4

cd csrc/layer_norm
pip install .
cd ../csrc/rotary
pip install .

cd /envs/mach-1/sources
 
git clone https://github.com/HazyResearch/flash-fft-conv.git
cd flash-fft-conv/csrc/flashfftconv
python setup.py install
cd ../..
python setup.py install

cd /envs/mach-1

pip install triton
pip install evo-model
pip install transformers tokenizers accelerate datasets evaluate
pip install wandb
pip install scipy