# ==== Model settings ====
# adaptation {finetune,lp}
#!/bin/bash

CSV=experiments.csv

tail -n +2 $CSV | while IFS=',' read MODEL MODEL_ARCH FINETUNE DATASET NUM_CLASS SEED
do
  name=$(basename $DATASET)_${FINETUNE%%.*}

  echo "🚀 $MODEL | $MODEL_ARCH | $FINETUNE | $DATASET | $NUM_CLASS | $SEED"

ADAPTATION="finetune"

# ==== Data settings ====
# change the dataset name and corresponding class number
DATASET="$DATASET"
DATA_PATH="./data/5_fold_${DATASET}/${DATASET}_seed${SEED}"
NUM_CLASS="${NUM_CLASS}"
task="${MODEL_ARCH}_${FINETUNE}_${DATASET}_seed${SEED}_${ADAPTATION}"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=48765 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size 24 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${task}" \
  --adaptation "${ADAPTATION}"
done



