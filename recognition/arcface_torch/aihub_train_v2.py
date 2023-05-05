import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from aihub_dataloader import aihub_dataloader
from facenet.validate_aihub import validate_aihub

assert (
    torch.__version__ >= "1.12.0"
), "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main():
    # get config
    config_path = "configs/aihub_r50_onegpu"
    cfg = get_config(config_path)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb

        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key, relogin=True)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = (
            run_name
            if cfg.suffix_run_name is None
            else run_name + f"_{cfg.suffix_run_name}"
        )
        try:
            wandb_logger = (
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    sync_tensorboard=True,
                    resume=cfg.wandb_resume,
                    name=run_name,
                    # notes=cfg.notes,
                )
                if rank == 0 or cfg.wandb_log_all
                else None
            )
            if wandb_logger:
                # wandb_logger.config.update(cfg)
                pass
            # Hyper parameters from wandb
            margin_list = wandb.config.margin_list
            network = wandb.config.network
            embedding_size = wandb.config.embedding_size
            sample_rate = wandb.config.sample_rate
            momentum = wandb.config.momentum  # SGD only
            weight_decay = wandb.config.weight_decay
            lr = wandb.config.lr
            dropout = wandb.config.dropout
            num_epoch = wandb.config.num_epoch
            optimizer = wandb.config.optimizer

        except Exception as e:
            print(
                "WandB Data (Entity and Project name) must be provided in config file (base.py)."
            )
            print(f"Config Error: {e}")

    train_loader = get_dataloader(
        cfg.rec, local_rank, cfg.batch_size, cfg.dali, cfg.seed, cfg.num_workers
    )

    backbone = get_model(
        network, dropout=dropout, fp16=cfg.fp16, num_features=embedding_size
    ).cuda()

    if cfg.pretrained:
        model_weights = torch.load(
            f"/home/jupyter/face/utils/model/arcface/{network}/backbone.pth"
        )
        backbone.load_state_dict(model_weights)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True,
    )

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        margin_list[0],
        margin_list[1],
        margin_list[2],
        cfg.interclass_filtering_threshold,
    )

    if optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, embedding_size, cfg.num_classes, sample_rate, cfg.fp16
        )
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, embedding_size, cfg.num_classes, sample_rate, cfg.fp16
        )
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1,
    )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(
            os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        )
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets,
        rec_prefix=cfg.rec,
        summary_writer=summary_writer,
        wandb_logger=wandb_logger,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
                    lr_scheduler.step()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log(
                        {
                            "Loss/Step Loss": loss.item(),
                            "Loss/Train Loss": loss_am.avg,
                            "Process/Step": global_step,
                            "Process/Epoch": epoch,
                            "Learning rate": lr_scheduler.get_last_lr()[0],
                        }
                    )

                loss_am.update(loss.item(), 1)
                callback_logging(
                    global_step,
                    loss_am,
                    epoch,
                    cfg.fp16,
                    lr_scheduler.get_last_lr()[0],
                    amp,
                )

                if global_step % cfg.verbose == 0 and global_step > 0:
                    best_distances, (
                        accuracy,
                        precision,
                        recall,
                        roc_auc,
                        tar,
                        far,
                    ) = validate_aihub(backbone, aihub_dataloader, network, 0)

                    val_accuracy = np.mean(accuracy)
                    val_precision = np.mean(precision)
                    val_recall = np.mean(recall)
                    val_best_distances = np.mean(best_distances)
                    val_tar = np.mean(tar)
                    val_far = np.mean(far)

                    if wandb_logger:
                        wandb_logger.log(
                            {
                                "val_accuracy": val_accuracy,
                                "val_precision": val_precision,
                                "val_recall": val_recall,
                                "val_best_distances": val_best_distances,
                                "val_tar": val_tar,
                                "val_far": val_far,
                                "roc_auc": roc_auc,
                            }
                        )

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(
                checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
            )

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type="model")
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type="model")
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    sweep_configuration = {
        "name": "ArcFace Hyperparameter Optimization",
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [0.2, 0.02, 0.002]},
            "margin_list": {
                "values": [
                    (1.0, 0.5, 0.0),
                    # (0.8, 0.4, 0.0),
                    # (1.2, 0.6, 0.0)
                ]
            },
            "network": {"values": ["r50"]},
            "embedding_size": {"values": [512]},
            "sample_rate": {"min": 0.7, "max": 1.0, "distribution": "uniform"},
            "momentum": {
                "values": [
                    0.9,
                    0.99,
                    0.999,
                ]
            },
            "weight_decay": {
                "values": [
                    5e-5,
                    5e-4,
                    5e-3,
                ]
            },
            "dropout": {"min": 0.0, "max": 0.5, "distribution": "uniform"},
            "num_epoch": {"values": [1, 2, 5]},
            "optimizer": {
                "values": [
                    "sgd",
                    # "adamw",
                ]
            },
        },
        "early_terminate": {"type": "hyperband", "min_iter": 1},
    }

    torch.backends.cudnn.benchmark = True
    sweep_id = wandb.sweep(
        sweep_configuration,
        entity="jongphago",
        project="arcface-hparams-optimization-aihub-age",
    )
    wandb.agent(sweep_id, main)
