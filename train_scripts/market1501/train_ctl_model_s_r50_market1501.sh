SEQ_NAME="U19_SKV_MIL_08_01_2022_1st_period_synced_1min"
EXPERIMENT_NAME="cutom_triplet_loss"
DATE="29_04_22"

DATASET_PATH="./data/"$SEQ_NAME"/"
OUTPUT_DIR="./logs/"$SEQ_NAME"/"$EXPERIMENT_NAME"/"$DATE

#######################################################################################
# Do not edit
#######################################################################################

python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES "frame_triplets_dataset" \
DATASETS.ROOT_DIR $DATASET_PATH \
SOLVER.IMS_PER_BATCH 4 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR $OUTPUT_DIR \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False