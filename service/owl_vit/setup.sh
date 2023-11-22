mkdir ./big_vision
git clone https://github.com/google-research/big_vision.git ./big_vision
python -m pip install -r ./big_vision/big_vision/requirements.txt

mkdir ./scenic
git clone https://github.com/google-research/scenic.git ./scenic
python -m pip install ./scenic
python -m pip install -r ./scenic/scenic/projects/owl_vit/requirements.txt

# git clone https://github.com/google-research/scenic.git
# python -m pip install -q ./scenic
# python -m pip install -r ./scenic/scenic/projects/owl_vit/requirements.txt

# # Also install big_vision, which is needed for the mask head:
# mkdir ./big_vision
# git clone https://github.com/google-research/big_vision.git ./big_vision
# python -m pip install -r ./big_vision/big_vision/requirements.txt
# echo "Done."

# Manual fix
pip install ott-jax==0.3.1