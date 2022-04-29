SEQ_NAME="U19_SKV_MIL_08_01_2022_1st_period_synced_1min"
EXPERIMENT_NAME="tiny_step_single_seq"
DATE="28_04_22"

DATASET_PATH="./data/"$SEQ_NAME"_market1501_CLUSTER_0"
OUTPUT_DIR="./logs/"$SEQ_NAME"/"$EXPERIMENT_NAME"/"$DATE

#######################################################################################
# Do not edit
#######################################################################################

python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES "custom_market_dataset" \
DATASETS.ROOT_DIR $DATASET_PATH \
SOLVER.IMS_PER_BATCH 32 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR $OUTPUT_DIR \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False