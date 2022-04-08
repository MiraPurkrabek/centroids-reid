python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'custom_market_dataset' \
DATASETS.ROOT_DIR './data/' \
SOLVER.IMS_PER_BATCH 32 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/custom_market_dataset_unique/256_resnet50/07_04_2022' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False