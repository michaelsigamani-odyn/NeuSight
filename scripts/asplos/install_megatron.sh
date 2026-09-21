pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v1.11
pip install accelerate==1.0.0
git clone https://github.com/NVIDIA/Megatron-LM
cd Megatron-LM
git checkout core_r0.5.0
pip install --no-use-pep517 -e .
