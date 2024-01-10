import torch
import numpy as np
import os
import json
import wandb
from collections import OrderedDict
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer
from torch import nn


class ModuleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def calculate_metrics(predictions, labels, task='multiclass'):
    assert len(predictions) == len(labels)
    metrics = {
        'accuracy': accuracy_score(labels, np.argmax(predictions, axis=-1)),
        'precision': precision_score(labels, np.argmax(predictions, axis=-1), average=task, zero_division=0),
        'recall': recall_score(labels, np.argmax(predictions, axis=-1), average=task, zero_division=0),
        'f1': f1_score(labels, np.argmax(predictions, axis=-1), average=task, zero_division=0)
    }
    return metrics

def evaluate(model, eval_loader, device, args):
    model.eval()
    all_predictions = []
    all_targets = []

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["true_label"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        predictions = torch.softmax(outputs.logits, dim=-1).cpu().detach().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(targets.cpu().numpy())

    metrics = calculate_metrics(all_predictions, all_targets)
    results_string = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    print(f"Evaluation Results ({args.evaluation_metric}={args.eval_threshold:.4f}) : {results_string}\n")

    if args.use_wandb:
        wandb.log({'val/' + k: v for k, v in metrics.items()})

def save_best_checkpoint(state_dict, global_step, epoch, args, logger, prefix='best'):
    """Save the best checkpoint"""
    is_new_best = False
    checkpoint_dir = os.path.join(args.output_dir, f'{prefix}_models')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_{global_step}.bin')
    model_to_save = state_dict.module if isinstance(state_dict, ModuleWrapper) else state_dict

    old_best = OrderedDict()
    try:
        old_best.update(torch.load(checkpoint_path))
    except FileNotFoundError:
        pass

    curr_metrics = {"epoch": epoch, "step": global_step}
    for metric_key in ["loss", "acc"]:
        old_best[metric_key] = old_best.get(metric_key, float('inf'))
        curr_metrics[metric_key] = state_dict.pop(metric_key)
        is_new_best |= curr_metrics[metric_key] < old_best[metric_key]

    if is_new_best:
        print(f"=> saving new best {prefix} model (epoch = {epoch}, step = {global_step})")
        torch.save(model_to_save, checkpoint_path)
        logger.info(f"=> saving new best {prefix} model (epoch = {epoch}, step = {global_step})\n")

    for k, v in old_best.items():
        state_dict[k] = v