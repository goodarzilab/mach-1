conda create -p /envs/mach-1 -y
conda activate /envs/mach-1

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install packaging
pip install ninja

mkdir /envs/mach-1/sources

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention 
python setup.py install
cd csrc/rotary 
pip install .
cd ../layer_norm 
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