#!/usr/bin/env bash
# ==== Model settings ====
# adaptation {finetune,lp}

port=$1

test_models=(
    # Pixio
    # "our_dis_pixio/pixio_vitl16_epoch-100.pth"
    # "our_dis_pixio/pixio_vitl16_epoch-120.pth"
    # "our_dis_pixio/pixio_vitl16_epoch-140.pth"
    # "our_dis_pixio/pixio_vitl16_epoch-160.pth"
    # "our_pixio/epoch-20.pth"
    # "our_pixio/epoch-40.pth"
    # "our_pixio/epoch-60.pth"
    # "our_pixio/epoch-80.pth"
    # "our_dis_pixio/epoch-20.pth"
    # "our_dis_pixio/epoch-40.pth"
    # "our_dis_pixio/epoch-60.pth"
    # "our_dis_pixio/epoch-80.pth"

    # Dinov3
    # "our_dinov3/epoch_29.pt"
    # "our_dinov3/epoch_50.pt"
    # "our_dinov3/epoch_74.pt"
    # "our_dinov3/exported_last.pt"
    # "our_dinov3/185199_keep_vitl.pth"
    # "our_dinov3/148159_keep_vitl.pth"
    # "our_dinov3/111119_keep_vitl.pth"
    # "our_dinov3/74079_keep_vitl.pth"
    # "our_dinov3/37039_keep_vitl.pth"

    # Baselines
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
    "Glaucoma_fundus"
    "PAPILA"
    "Retina"
    "MIL"
    "SL"
)

declare -A MODEL=(
    # Pixio
    ["our_dis_pixio/pixio_vitl16_epoch-100.pth"]="Pixio"
    ["our_dis_pixio/pixio_vitl16_epoch-120.pth"]="Pixio"
    ["our_dis_pixio/pixio_vitl16_epoch-140.pth"]="Pixio"
    ["our_dis_pixio/pixio_vitl16_epoch-160.pth"]="Pixio"
    ["our_pixio/epoch-20.pth"]="Pixio"
    ["our_pixio/epoch-40.pth"]="Pixio"
    ["our_pixio/epoch-60.pth"]="Pixio"
    ["our_pixio/epoch-80.pth"]="Pixio"
    ["our_dis_pixio/epoch-20.pth"]="Pixio"
    ["our_dis_pixio/epoch-40.pth"]="Pixio"
    ["our_dis_pixio/epoch-60.pth"]="Pixio"
    ["our_dis_pixio/epoch-80.pth"]="Pixio"
    ["pixio_vitl16.pth"]="Pixio"

    # Dinov3
    ["our_dinov3/epoch_29.pt"]="Dinov3"
    ["our_dinov3/epoch_50.pt"]="Dinov3"
    ["our_dinov3/epoch_74.pt"]="Dinov3"
    ["our_dinov3/exported_last.pt"]="Dinov3"
    ["our_dinov3/185199_keep_vitl.pth"]="Dinov3"
    ["our_dinov3/148159_keep_vitl.pth"]="Dinov3"
    ["our_dinov3/111119_keep_vitl.pth"]="Dinov3"
    ["our_dinov3/74079_keep_vitl.pth"]="Dinov3"
    ["our_dinov3/37039_keep_vitl.pth"]="Dinov3"

    # Baselines
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
    # pixio
    ["our_dis_pixio/pixio_vitl16_epoch-100.pth"]="Pixio"
    ["our_dis_pixio/pixio_vitl16_epoch-120.pth"]="Pixio"
    ["our_dis_pixio/pixio_vitl16_epoch-140.pth"]="Pixio"
    ["our_dis_pixio/pixio_vitl16_epoch-160.pth"]="Pixio"
    ["our_pixio/epoch-20.pth"]="Pixio"
    ["our_pixio/epoch-40.pth"]="Pixio"
    ["our_pixio/epoch-60.pth"]="Pixio"
    ["our_pixio/epoch-80.pth"]="Pixio"
    ["our_dis_pixio/epoch-20.pth"]="Pixio"
    ["our_dis_pixio/epoch-40.pth"]="Pixio"
    ["our_dis_pixio/epoch-60.pth"]="Pixio"
    ["our_dis_pixio/epoch-80.pth"]="Pixio"
    ["pixio_vitl16.pth"]="Pixio"

    # dinov3
    ["our_dinov3/epoch_29.pt"]="dinov3_vitl16"
    ["our_dinov3/epoch_50.pt"]="dinov3_vitl16"
    ["our_dinov3/epoch_74.pt"]="dinov3_vitl16"
    ["our_dinov3/exported_last.pt"]="dinov3_vitl16"
    ["our_dinov3/185199_keep_vitl.pth"]="dinov3_vitl16"
    ["our_dinov3/148159_keep_vitl.pth"]="dinov3_vitl16"
    ["our_dinov3/111119_keep_vitl.pth"]="dinov3_vitl16"
    ["our_dinov3/74079_keep_vitl.pth"]="dinov3_vitl16"
    ["our_dinov3/37039_keep_vitl.pth"]="dinov3_vitl16"

    # Baselines
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
    ["Glaucoma_fundus"]="3"
    ["PAPILA"]="3"
    ["Retina"]="4"
    ["MIL"]="2"
    ["SL"]="2"
)

declare -A epochs=(
    # ["APTOS2019"]="50"
    # ["MESSIDOR2"]="50"
    # ["IDRiD_data"]="50"
    # ["Glaucoma_fundus"]="50"
    # ["PAPILA"]="50"
    # ["Retina"]="50"
    # ["MIL"]="100"
    # ["SL"]="100"
    
    ["APTOS2019"]="2"
    ["MESSIDOR2"]="2"
    ["IDRiD_data"]="2"
    ["Glaucoma_fundus"]="2"
    ["PAPILA"]="2"
    ["Retina"]="2"
    ["MIL"]="2"
    ["SL"]="2"
)


for CUR_MODEL in "${test_models[@]}"; do
    for CUR_DATASET in "${test_datasets[@]}"; do
        for FOLD in {0..4}; do
            MODEL_NAME="${MODEL["$CUR_MODEL"]}"
            MODEL_ARCH="${ARCH["$CUR_MODEL"]}"
            NUM_CLASS="${CLASS["$CUR_DATASET"]}"
            NUM_EPOCHS="${epochs["$CUR_DATASET"]}"
            

            echo "🚀 $MODEL_NAME | $MODEL_ARCH | $CUR_MODEL | $CUR_DATASET | $NUM_CLASS | $FOLD"
            # DATA_PATH="./data/${CUR_DATASET}"
            DATA_PATH="./data/5_fold_${CUR_DATASET}/${CUR_DATASET}_seed42_fold${FOLD}"
            # task="${MODEL_ARCH}_${CUR_MODEL}_${CUR_DATASET}_FOLD${FOLD}_origin_finetune"
            task="${MODEL_ARCH}_${CUR_MODEL}_${CUR_DATASET}_FOLD${FOLD}_finetune"

            torchrun --nproc_per_node=1 --master_port="${port}" main_finetune.py \
              --model "${MODEL_NAME}" \
              --model_arch "${MODEL_ARCH}" \
              --finetune "${CUR_MODEL}" \
              --savemodel \
              --global_pool \
              --batch_size 24 \
              --world_size 1 \
              --epochs "${NUM_EPOCHS}" \
              --nb_classes "${NUM_CLASS}" \
              --data_path "${DATA_PATH}" \
              --output_dir "./test" \
              --input_size 224 \
              --task "${task}" \
              --adaptation finetune
        done
    done
done
