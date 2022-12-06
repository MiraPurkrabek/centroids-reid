# Edit this manually
SEQ_NAME="U19_SKV_MIL_08_01_2022_1st_period_synced"
# SEQ_NAME="WA_SKV_SPA_04_03_2022_1st_period_synced_1min"
MODEL_SEQ_NAME="original"
# MODEL_SEQ_NAME="U19_SKV_MIL_08_01_2022_1st_period_synced_1min"
EXPERIMENT_NAME="resnet50_market1501_pretrained"
# EXPERIMENT_NAME="custom_triplet_loss"
EPOCH=120

DATASET_PATH="data/"$SEQ_NAME"/"$SEQ_NAME"_custom_GEOM/"
EMBEDDINGS_OUTPUT="./embeddings/"$SEQ_NAME"/"$MODEL_SEQ_NAME"_"$EXPERIMENT_NAME"_"$EPOCH
MODEL_PATH="./models/"$MODEL_SEQ_NAME"/"$MODEL_SEQ_NAME"_"$EXPERIMENT_NAME"_"$EPOCH".pth"

#######################################################################################
# Do not edit
#######################################################################################

python3 inference/create_embeddings.py \
--config_file="configs/experiments/"$EXPERIMENT_NAME".yml" \
DATASETS.ROOT_DIR $DATASET_PATH \
OUTPUT_DIR $EMBEDDINGS_OUTPUT \
MODEL.PRETRAIN_PATH $MODEL_PATH \
TEST.IMS_PER_BATCH 256 \
TEST.ONLY_TEST True 
