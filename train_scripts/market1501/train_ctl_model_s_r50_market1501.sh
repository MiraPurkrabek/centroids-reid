SEQ_NAME="U19_SKV_MIL_08_01_2022_2nd_period_synced_3shifts"
SEQ_NAME="MA_SKV_LIB_20_11_2022_1st_period_synced_manual"

DATASET_PATH="./data/"$SEQ_NAME"/"
DATASET_PATH="/datagrid/personal/purkrmir/data/centroids_learning/"$SEQ_NAME"/"

EXPERIMENT_NAME="merged_datasets"
EXPERIMENT_NAME="NUMBERS"
EXPERIMENT_NAME="floorball_manual_market"
EXPERIMENT_NAME="patches_market"
EXPERIMENT_NAME="GPHMER_texture_SMPL"
EXPERIMENT_NAME="patches_market_separate"

DATE=$(date +"%m_%d_%y")
TIME=$(date +"%H_%M")

OUTPUT_DIR="./logs/"$SEQ_NAME"/"$EXPERIMENT_NAME"/backNumber/"$DATE

#######################################################################################
# Do not edit
#######################################################################################

# Prepare directory
mkdir -p $OUTPUT_DIR

# Run the training with output both to the log file and stdout
python3 train_ctl_model.py \
--config_file="configs/experiments/"$EXPERIMENT_NAME".yml" \
DATASETS.ROOT_DIR $DATASET_PATH \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "logs/MA_SKV_LIB_20_11_2022_1st_period_synced_manual/patches_market_separate/backNumber/12_14_22/train_ctl_model/version_0/auto_checkpoints/checkpoint_110.pth" \
OUTPUT_DIR $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR"/"$TIME".log"

# MODEL.PRETRAIN_PATH "models/original/original_resnet50_market1501_pretrained_120.pth" \