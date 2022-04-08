python3 train_ctl_model.py \
--config_file="configs/256_resnet50_test.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'custom_market_dataset' \
DATASETS.ROOT_DIR './data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/custom_market_dataset_unique/256_resnet50/08_04_2022_retest' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "./logs/custom_market_dataset_unique/256_resnet50/07_04_2022/train_ctl_model/version_2/auto_checkpoints/checkpoint_119.pth"
# OUTPUT_DIR './logs/market1501/256_resnet50' \
# MODEL.PRETRAIN_PATH "logs/market1501/256_resnet50/train_ctl_model/version_16/checkpoints/epoch=79.ckpt"
