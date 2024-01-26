python ./src/evaluate.py \
    --params_file "./example_confs/standard_vqvae_cb1024.yaml" \
    --dataloader ffcv \
    --dataset_path "home_datasets/" \
    --batch_size 64 \
    --seed 42 \
    --loading_path "home_models/standard_vqvae_cb1024/last.ckpt" \
    --workers 8