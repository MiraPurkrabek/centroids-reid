SEQ_NAME="U19_SKV_MIL_08_01_2022_2nd_period_synced_3shifts"
EXPERIMENT_NAME="GPHMER_texture_SMPL"
EXPERIMENT_NAME="merged_datasets"
EXPERIMENT_NAME="NUMBERS"
DATE=$(date +"%m_%d_%y")
TIME=$(date +"%H_%M")

DATASET_PATH="./data/"$SEQ_NAME"/"
OUTPUT_DIR="./logs/"$SEQ_NAME"/"$EXPERIMENT_NAME"/"$DATE

#######################################################################################
# Do not edit
#######################################################################################

# Prepare directory
mkdir -p $OUTPUT_DIR

# Run the training with output both to the log file and stdout
python3 train_ctl_model.py \
--config_file="configs/experiments/"$EXPERIMENT_NAME".yml" \
DATASETS.ROOT_DIR $DATASET_PATH \
OUTPUT_DIR $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR"/"$TIME".log"
