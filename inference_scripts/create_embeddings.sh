
python3 inference/create_embeddings.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.ROOT_DIR 'data/custom_market_dataset_unique/query/' \
TEST.IMS_PER_BATCH 128 \
OUTPUT_DIR 'data/embeddings/custom_market_dataset_unique/query/' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "./logs/custom_market_dataset_unique/256_resnet50/07_04_2022/train_ctl_model/version_2/auto_checkpoints/checkpoint_119.pth"