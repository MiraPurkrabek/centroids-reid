# Edit this manually
SEQ_NAME="U19_SKV_MIL_08_01_2022_1st_period_synced_1min"
EXPERIMENT_NAME="tint_steps_single_seq"

DATASET_PATH="data/"$SEQ_NAME"_custom_GEOM/"
EMBEDDINGS_OUTPUT="./embeddings/"$SEQ_NAME"/"$EXPERIMENT_NAME
MODEL_PATH="./models/"$SEQ_NAME"/"$SEQ_NAME"_market1501_CLUSTER_0.pth"

# MODEL.PRETRAIN_PATH "./logs/custom_market_dataset_unique/256_resnet50/23_04_2022/train_ctl_model/version_4/auto_checkpoints/checkpoint_119.pth"
# MODEL.PRETRAIN_PATH "./logs/custom_market_dataset_tiny_steps_seq_1_round2/256_resnet50/26_04_2022/train_ctl_model/version_0/auto_checkpoints/checkpoint_119.pth"

#######################################################################################
# Do not edit
#######################################################################################

python3 inference/create_embeddings.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.ROOT_DIR $DATASET_PATH \
TEST.IMS_PER_BATCH 256 \
OUTPUT_DIR $EMBEDDINGS_OUTPUT \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH $MODEL_PATH
