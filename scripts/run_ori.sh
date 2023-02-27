source ~/.bashrc
conda activate hypernerf

python train.py --base_folder experiments/ori_scratch \
                --gin_bindings="data_dir='datasets/vrig-chicken'" \
                --gin_configs configs/hypernerf_vrig_ds_2d.gin
