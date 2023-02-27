source ~/.bashrc
conda activate hypernerf

python train.py --base_folder experiments/scratch \
                --gin_bindings="data_dir='datasets/formatted_checkers'" \
                --gin_configs configs/dev.gin
