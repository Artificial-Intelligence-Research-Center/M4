#!/bin/bash
# Waits until the (Exclusive_Process) GPU is actually allocatable, then launches script.py.
# The GPU is currently held by an external process (another container) in Exclusive_Process
# compute mode, so we must poll with a real CUDA allocation, not just memory.used.
set -u
cd /home/jovyan/M4
source /opt/conda/etc/profile.d/conda.sh
conda activate M4

LOG=/home/jovyan/M4/fundus_baseline_finetune_run.log
mkdir -p /home/jovyan/M4/fundus_baseline_finetune
echo "[launcher] waiting for GPU to become allocatable ..." | tee -a "$LOG"

while true; do
    if CUDA_VISIBLE_DEVICES=0 python -c "import torch; torch.randn(5).cuda()" 2>/dev/null; then
        echo "[launcher] GPU is free at $(date -Iseconds). Launching script.py" | tee -a "$LOG"
        break
    fi
    sleep 60
done

# Full sweep: 3 models x 2 datasets x 5 folds x 2 param-sets = 60 jobs, 50 epochs each.
python script.py >> "$LOG" 2>&1
echo "[launcher] script.py exited with code $? at $(date -Iseconds)" | tee -a "$LOG"
