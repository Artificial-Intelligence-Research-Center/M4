#!/usr/bin/env bash
# ==== Model settings ====
# adaptation {finetune,lp}

port=$1

test_models=(
    "test.pth"
    # "our_pixio/epoch-20.pth"
    # "our_pixio/epoch-40.pth"
    # "our_pixio/epoch-60.pth"
    # "our_pixio/epoch-80.pth"
    # "pixio_vitl16.pth"
    # "vit_large_patch16_224.pth"
    # "mae_pretrain_vit_large.pth"
    # "RETFound_mae_natureCFP"
    # "RETFound_mae_meh"
    # "RETFound_mae_shanghai"
    # "RETFound_dinov2_meh"
    # "RETFound_dinov2_shanghai"
    # "dinov2_vitl14_pretrain.pth"
    # "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

test_datasets=(
  "APTOS2019"
  "MESSIDOR2"
  "IDRiD_data"
  "Glaucoma_fundas"
  "PAPILA"
  "Retina"
)

declare -A MODEL=(
    ["test.pth"]="MAE"
    # ["our_pixio/epoch-20.pth"]="MAE"
    # ["our_pixio/epoch-40.pth"]="MAE"
    # ["our_pixio/epoch-60.pth"]="MAE"
    # ["our_pixio/epoch-80.pth"]="MAE"
    ["pixio_vitl16.pth"]="MAE"
    ["vit_large_patch16_224.pth"]="MAE"
    ["mae_pretrain_vit_large.pth"]="MAE"
    ["RETFound_mae_natureCFP"]="RETFound_mae"
    ["RETFound_mae_meh"]="RETFound_mae"
    ["RETFound_mae_shanghai"]="RETFound_mae"
    ["RETFound_dinov2_meh"]="RETFound_dinov2"
    ["RETFound_dinov2_shanghai"]="RETFound_dinov2"
    ["dinov2_vitl14_pretrain.pth"]="Dinov2"
    ["dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"]="Dinov3"
)

# finetune -> model_arch
declare -A ARCH=(
    ["test.pth"]="MAE"
    # ["our_pixio/epoch-20.pth"]="MAE"
    # ["our_pixio/epoch-40.pth"]="MAE"
    # ["our_pixio/epoch-60.pth"]="MAE"
    # ["our_pixio/epoch-80.pth"]="MAE"
    ["pixio_vitl16.pth"]="MAE"
    ["vit_large_patch16_224.pth"]="MAE"
    ["mae_pretrain_vit_large.pth"]="MAE"
    ["RETFound_mae_natureCFP"]="retfound_mae"
    ["RETFound_mae_meh"]="retfound_mae"
    ["RETFound_mae_shanghai"]="retfound_mae"
    ["RETFound_dinov2_meh"]="retfound_dinov2"
    ["RETFound_dinov2_shanghai"]="retfound_dinov2"
    ["dinov2_vitl14_pretrain.pth"]="dinov2_vitl14"
    ["dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"]="dinov3_vitl16"
)

declare -A CLASS=(
    ["APTOS2019"]="5"
    ["MESSIDOR2"]="5"
    ["IDRiD_data"]="5"
    ["Glaucoma_fundas"]="3"
    ["PAPILA"]="3"
    ["Retina"]="4"
)


for CUR_MODEL in "${test_models[@]}"; do
    for CUR_DATASET in "${test_datasets[@]}"; do
        for FOLD in {0..4}; do
            MODEL_NAME="${MODEL["$CUR_MODEL"]}"
            MODEL_ARCH="${ARCH["$CUR_MODEL"]}"
            NUM_CLASS="${CLASS["$CUR_DATASET"]}"
            

            echo "🚀 $MODEL_NAME | $MODEL_ARCH | $CUR_MODEL | $CUR_DATASET | $NUM_CLASS | $FOLD"
            DATA_PATH="./data/5_fold_${CUR_DATASET}/${CUR_DATASET}_seed42_fold${FOLD}"
            task="${MODEL_ARCH}_${CUR_MODEL}_${CUR_DATASET}_FOLD${FOLD}_finetune"

            torchrun --nproc_per_node=1 --master_port="${port}" main_finetune.py \
              --model "${MODEL_NAME}" \
              --model_arch "${MODEL_ARCH}" \
              --finetune "${CUR_MODEL}" \
              --savemodel \
              --global_pool \
              --batch_size 24 \
              --world_size 1 \
              --epochs 50 \
              --nb_classes "${NUM_CLASS}" \
              --data_path "${DATA_PATH}" \
              --input_size 224 \
              --task "${task}" \
              --adaptation finetune
        done
    done
done
