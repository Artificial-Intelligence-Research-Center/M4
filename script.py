import subprocess
import time
import os
from multiprocessing import Process, Semaphore, Queue
import GPUtil

# ==== 配置設定 ====
TEST_FOLDERS = [
    # "gas_loss_exp",
    "baseline_models",
    # "gastro_dinov3"
    # "our_dinov3"
    "gastroscopy_baseline",
    # "our_pixio_dis"
    "dinov2_base",
    # "dinov3_cs"
    # "dinov3_0414"
    # "sft_default_param",
    # "sft_mae_param",
    # "sft_sft_param",
    # "imagenet_sft_default_param",
    # "imagenet_sft_mae_param",
    # "imagenet_sft_sft_param",
    # "sft_10_20_mae"
    # "retfound_sft_eval_models"
    # "dinov3_0429",
    # "only_vit",
    # "sft_fundus_mae_param",
    # "sft_imagenet2fundus_models/eval_models",
    # "experiment_redo"
    # "redo_SFT_finetune_0601"
    # "dinov2_vitb14_gastronet_0609"
]

TEST_DATASETS = [
    # "APTOS2019",
    # "MESSIDOR2",
    # "IDRiD_data",
    # "Glaucoma_fundus",
    # "PAPILA",
    # "Retina",
    # "MIL",
    "SL",
    # "HK",
]


CLASS_MAP = {
    "APTOS2019": "5", "MESSIDOR2": "5", "IDRiD_data": "5",
    "Glaucoma_fundus": "3", "PAPILA": "3", "Retina": "4",
    "MIL": "2", "SL": "2", "HK": "23",
}

def get_free_gpus():
    """ 取得目前顯存佔用低於 10% 的 GPU ID """
    return GPUtil.getAvailable(order='first', limit=8, maxLoad=0.1, maxMemory=0.1)

def run_experiment(gpu_id, cmd, task_name, output_dir):
    """ 執行單一實驗任務 """
    # 設定該進程專用的 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🔥 開始任務: {task_name} | 使用 GPU: {gpu_id}")
    
    with open(output_dir, "w") as f:
        process = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=f)
        process.wait()
    
    print(f"✅ 任務完成: {task_name} (GPU {gpu_id})")

def worker(task_queue, gpu_semaphore):
    """ 監聽隊列並分配 GPU """
    while True:
        # 從隊列拿任務
        task = task_queue.get()
        if task is None: break
        
        cmd, task_name = task
        
        # 等待有可用的 GPU 位置
        gpu_semaphore.acquire()
        
        # 尋找當前真正空的 GPU
        available_gpus = []
        while not available_gpus:
            available_gpus = get_free_gpus()
            if not available_gpus:
                time.sleep(30) # 如果沒位子，等 30 秒再檢查
        
        target_gpu = available_gpus[0]
        
        # 開啟子進程跑實驗
        p = Process(target=run_experiment, args=(target_gpu, cmd, task_name))
        p.start()
        
        # 這裡需要監控 p 結束後釋放 semaphore，
        # 簡單做法是在 run_experiment 結束後手動釋放（需傳入 semaphore）
        # 或是在主程式維護一個 thread pools

def get_model_info(model_path):
    """ 根據模型路徑解析出模型名稱、架構等資訊 """
    if "retfound_mae" in model_path.lower():
        return "RETFound_mae", "retfound_mae"
    elif "retfound_dinov2" in model_path.lower():
        return "RETFound_dinov2", "retfound_dinov2"
    elif "gastronet_dinov2" in model_path.lower():
        return "GastroNet", "gastro_dinov2"
    elif "dinov3" in model_path.lower():
        return "Dinov3", "dinov3_vitl16"
    elif "dinov2_vitb14" in model_path.lower():
        return "Dinov2", "dinov2_vitb14"
    elif "dinov2" in model_path.lower():
        return "Dinov2", "dinov2_vitl14"
    elif "pixio" in model_path.lower():
        return "Pixio", "Pixio"
    elif "mae" in model_path.lower():
        return "MAE", "MAE"
    elif "vit_large_patch16" in model_path.lower():
        return "SL_VIT", "SL_VIT"
    else:
        raise ValueError(f"無法解析模型資訊: {model_path}")

if __name__ == "__main__":
    # output_dir = "experiment_redo"
    # output_dir = "gas_loss_exp"
    # output_dir = "test"
    # output_dir = "redo_SFT_finetune_0601"
    # output_dir = "dinov2_vitb14_gastronet_0609"
    output_dir = "gastroscopy_baseline_finetune_0612"

    # 0. 先收集所有模型檔案
    test_models = []
    # test_models.extend(
    #     [
    #         'RETFound_mae_natureCFP',
    #         'RETFound_mae_meh',
    #         'RETFound_mae_shanghai',
    #         'RETFound_dinov2_meh',
    #         'RETFound_dinov2_shanghai'
    #     ]
    # )
    for path in TEST_FOLDERS:
        model_ckpts = [f for f in os.listdir(path) if f.endswith((".pth", ".pt"))]
        test_models.extend([os.path.join(path, ckpt) for ckpt in model_ckpts])

    tasks = []
    # 1. 產生所有任務指令
    for cur_model in test_models:
        for cur_dataset in TEST_DATASETS:
            for fold in [0, 1, 2, 3, 4]:
            # for fold in [0]:
                model_name, model_arch = get_model_info(cur_model)
                num_class = CLASS_MAP[cur_dataset]
                
                data_path = f"./data/5_fold_{cur_dataset}/{cur_dataset}_seed42_fold{fold}"
                task_id = f"{model_arch}_{cur_model.replace('/', '_')}_{cur_dataset}_FOLD{fold}_finetune"
                
                # 組合指令 (注意：這裡不需要 --master_port，因為我們用 CUDA_VISIBLE_DEVICES 隔離)
                # Default
                cmd = (
                    f"python main_finetune.py "
                    f"--model {model_name} --model_arch {model_arch} --finetune {cur_model} "
                    f"--savemodel --global_pool --batch_size 24 --epochs 50 "
                    f"--nb_classes {num_class} --data_path {data_path} "
                    f"--output_dir {output_dir}/default_param --input_size 224 --task {task_id} --adaptation finetune"
                )
                if not os.path.exists(f"{output_dir}/default_param/{task_id}"):
                    print(f"加入任務: {task_id}")
                    tasks.append((cmd, task_id))

                # papers
                # cmd = (
                #     f"python main_finetune.py "
                #     f"--model {model_name} --model_arch {model_arch} --finetune {cur_model} "
                #     f"--savemodel --global_pool --batch_size 24 --epochs 50 --blr 5e-4 "
                #     f"--nb_classes {num_class} --data_path {data_path} "
                #     f"--output_dir {output_dir}/paper_param --input_size 224 --task {task_id} --adaptation finetune"
                # )
                # if not os.path.exists(f"{output_dir}/paper_param/{task_id}"):
                #     print(f"加入任務: {task_id}")
                #     tasks.append((cmd, task_id))

                # MAE
                cmd = (
                    f"python main_finetune.py "
                    f"--model {model_name} --model_arch {model_arch} --finetune {cur_model} "
                    f"--savemodel --global_pool --batch_size 32 --accum_iter 2 --drop_path 0.1 --epochs 50 "
                    f"--nb_classes {num_class} --data_path {data_path} --blr 1e-3 --layer_decay 0.75 "
                    f"--output_dir {output_dir}/mae_param --input_size 224 --task {task_id} --adaptation finetune"
                )
                if not os.path.exists(f"{output_dir}/mae_param/{task_id}"):
                    print(f"加入任務: {task_id}")
                    tasks.append((cmd, task_id))


    # 2. 自動偵測 GPU 數量
    all_gpus = GPUtil.getGPUs()
    num_gpus = len(all_gpus)
    print(f"系統偵測到 {num_gpus} 張顯卡。準備執行 {len(tasks)} 個任務...")

    # 3. 使用簡單的迴圈分配任務 (或進階的進程池)
    # 這裡示範簡易版：每張卡各跑一個任務，跑完補上
    active_processes = []
    
    count = 0
    while tasks or active_processes:
        # 移除已完成的進程
        active_processes = [p for p in active_processes if p[0].is_alive()]
        
        # 檢查哪些 GPU 閒置
        busy_gpus = [p[1] for p in active_processes]
        free_gpus = [g.id for g in GPUtil.getGPUs() if g.id not in busy_gpus and g.memoryUtil < 0.1]
        
        while free_gpus and tasks:
            gpu_to_use = free_gpus.pop(0)
            cmd, t_name = tasks.pop(0)
            
            if not os.path.exists(os.path.join(output_dir, 'logs')):
                os.makedirs(os.path.join(output_dir, 'logs'))
            p = Process(target=run_experiment, args=(gpu_to_use, cmd, t_name, os.path.join(output_dir, 'logs', f"log_{t_name}.txt")))
            p.start()
            print(f"分配任務: {t_name} 到 GPU {gpu_to_use} (剩餘任務數: {len(tasks)})")
            active_processes.append((p, gpu_to_use))
            
        time.sleep(10) # 每 10 秒檢查一次 GPU 狀態