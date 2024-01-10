import utils
import logging
import os
import torch
import time
import datetime
import pytz
import hydra
import omegaconf
from rich.console import Console
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter
from src.model import CodeLlamaForSequenceClassification
from src.data import HTMLDataset
from src.utils import calculate_metrics, evaluate, save_best_checkpoint
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)
console = Console()

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: omegaconf.OmegaConf):
    console.print("[bold blue]Start Finetuning!\n")

    device = torch.device(cfg.device)

    # Initialize model, tokenizer, and datasets
    model = CodeLlamaForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_classes
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset = HTMLDataset(
        cfg.train_csv_file, tokenizer, 'train', cfg.max_seq_len
    )
    dev_dataset = HTMLDataset(cfg.dev_csv_file, tokenizer, 'validation', cfg.max_seq_len)

    # Setup train and eval dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eva_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Initialize tensorboard writer
    tbwriter = SummaryWriter(cfg.tb_log_dir)

    # Train model
    global_step, best_epoch, best_step = 0, 0, 0
    early_stopping_counter = 0

    for epoch in range(1, cfg.num_train_epochs + 1):
        model.train()
        model.zero_grad()



    console.print("\nFinished Finetuning.")

if __name__ == "__main__":
    main()