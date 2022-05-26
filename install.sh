# Create the conda environment

conda env create --file environment.yml
eval "$(conda shell.bash hook)"
conda activate watch_it_move
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"

# Initialize the submodules

cd data_preprocess/zju/EasyMocap
python setup.py develop
# Patch lbs.py to expose a local variable we need use 
patch -p1 < ../diff.patch
cd ../../..

cd AdelaiDet
python setup.py build develop
cd ..

cd Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
