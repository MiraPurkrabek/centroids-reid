SEQ_NAME="MA_SKV_LIB_20_11_2022_1st_period_synced_manual"
EXPERIMENT_NAME="patches_custom_market"
DATE="14_12_22"

# DATASET_PATH="./data/"$SEQ_NAME"_market1501_CLUSTER_0"
# DATASET_PATH="/datagrid/personal/purkrmir/data/centroids_learning/MA_SKV_LIB_20_11_2022_1st_period_synced_manual/patches_market_by_parts/"
DATASET_PATH="/datagrid/personal/purkrmir/data/centroids_learning"
OUTPUT_DIR="./logs/"$SEQ_NAME"/"$EXPERIMENT_NAME"/"$DATE

#######################################################################################
# Do not edit
#######################################################################################

python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
OUTPUT_DIR $OUTPUT_DIR \
DATASETS.NAMES "custom_market_dataset" \
DATASETS.ROOT_DIR $DATASET_PATH \
SOLVER.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 0.00035 \
TEST.IMS_PER_BATCH 128 \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
REPRODUCIBLE_NUM_RUNS 1 \