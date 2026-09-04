import os
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched
from multilabel import evaluate_multilabel

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
    model_without_ddp=None,
    data_loader_val=None,
    evaluate_fn=None,
    num_class=None,
    iter_state=None,
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')

    save_iter = getattr(args, "save_iter", 0)
    iters_per_epoch = len(data_loader)

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if args.classification_type == "multi_label":
            targets = targets.float()
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if args.classification_type == "multi_label" and outputs.shape != targets.shape:
                raise ValueError(
                    f"Multi-label logits shape {tuple(outputs.shape)} does not match "
                    f"target shape {tuple(targets.shape)}"
                )
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        # ---- Evaluate & save checkpoint every N iterations ----
        if (
            save_iter > 0
            and (data_iter_step + 1) % save_iter == 0
        ):
            global_iter = epoch * iters_per_epoch + data_iter_step + 1
            print(f"[save_iter] Global iter {global_iter} (epoch {epoch}, step {data_iter_step + 1})")

            # --- evaluation ---
            if evaluate_fn is not None and data_loader_val is not None:
                model.eval()
                _, val_score, pred_outputs = evaluate_fn(
                    data_loader_val, model, device, args,
                    epoch=global_iter, mode="val",
                    num_class=num_class, log_writer=log_writer,
                )
                model.train(True)

                output_task_dir = os.path.join(args.output_dir, args.task)
                os.makedirs(output_task_dir, exist_ok=True)

                if args.SFT:
                    iter_csv = os.path.join(output_task_dir, f"predictions_iter{global_iter}.csv")
                    pred_outputs.to_csv(iter_csv, index=False, encoding="utf-8-sig")

                is_best_iter = (iter_state is not None) and (val_score > iter_state["max_score"])
                if is_best_iter:
                    iter_state["max_score"] = val_score
                    iter_state["best_iter"] = global_iter
                    best_csv = os.path.join(output_task_dir, "predictions_val.csv")
                    pred_outputs.to_csv(best_csv, index=False, encoding="utf-8-sig")
                    print(f"[save_iter] New best score {val_score:.4f} at iter {global_iter}")

                print(f"[save_iter] Best iter = {iter_state['best_iter'] if iter_state else '-'}, "
                      f"Best score = {iter_state['max_score'] if iter_state else '-'}")
            else:
                is_best_iter = False

            # --- save checkpoint ---
            if model_without_ddp is not None and getattr(args, "output_dir", None) and getattr(args, "savemodel", False):
                if is_best_iter:
                    # 覆寫 best checkpoint
                    misc.save_model(
                        args=args, epoch=epoch, model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        mode="iter", global_iter=global_iter,
                    )
                else:
                    misc.save_model(
                        args=args, epoch=epoch, model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        mode="iter", global_iter=global_iter,
                    )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    if args.classification_type == "multi_label":
        return evaluate_multilabel(
            data_loader, model, device, args, epoch, mode, num_class, log_writer
        )
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    paths = []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target, path = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2]
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)
        
        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())
        paths.extend(path)
    

    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    
    # score = (f1 + roc_auc + kappa) / 3
    score = roc_auc
    if log_writer:
        for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'score'],
                                       [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
    
    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n'
          f' Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
          f' Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}')
    
    metric_logger.synchronize_between_processes()

    # Prepare predictions for saving
    if isinstance(data_loader.dataset, torch.utils.data.Subset):
        classes = data_loader.dataset.dataset.classes
    else:
        classes = data_loader.dataset.classes
    
    # pred_softmax: list of [num_class]
    df = pd.DataFrame(pred_softmax, columns=[f"{cls}_score" for cls in classes])
    # 加上路徑
    df.insert(0, "image_path", paths)
    df["pred_label"] = pred_labels
    df["true_label"] = true_labels
    df["pred_class"] = [classes[i] for i in pred_labels]
    df["true_class"] = [classes[i] for i in true_labels]
    
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa'])
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa])
    
    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score, df
