source ~/.bashrc
conda activate hypernerf

python train.py --base_folder experiments/dev \
                --gin_bindings="data_dir='datasets/pick'" \
                --gin_configs configs/dev.gin
