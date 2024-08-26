import time
import wandb
import torch
import warnings
import logging

# surpress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)

import torch.distributed as dist

from copy import deepcopy
from datetime import timedelta
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union, Any

from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed

# TODO: use STDiT3 as DiT backbone?
# https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3
from opensora.models.stdit.stdit2 import STDiT2
from opensora.models.vae import VideoAutoencoderKL
from opensora.schedulers.iddpm import IDDPM
from opensora.utils.config_entity import TrainingConfig
from opensora.utils.model_helpers import calculate_weight_norm, push_to_device
from opensora.utils.lr_schedulers import ConstantWarmupLR, OneCycleScheduler
from opensora.utils.wandb_logging import write_sample, log_sample
from opensora.datasets.datasets import VariableVideoTextDataset, VariableNBAClipsDataset
from opensora.datasets.sampler import VariableVideoBatchSampler, VariableNBAClipsBatchSampler
from opensora.utils.sampler_entities import MicroBatch
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import (
    prepare_variable_dataloader,
)
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import (
    create_logger,
    load,
    model_sharding,
    record_model_param_shape,
    save,
)
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import (
    all_reduce_mean,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, update_ema

CAPTION_CHANNELS = 4096
MODEL_MAX_LENGTH = 200


def train(
    cfg: TrainingConfig,
    coordinator: DistCoordinator,
    logger: logging.Logger,
    vae: VideoAutoencoderKL,
    model: STDiT2,
    scheduler: IDDPM,
    start_epoch: int,
    dataloader: torch.utils.data.DataLoader,
    mask_generator: MaskGenerator,
    num_steps_per_epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    booster: Booster,
    optimizer: HybridAdam,
    lr_scheduler: OneCycleScheduler,
    ema: STDiT2,
    writer,
    exp_dir: str,
    ema_shape_dict: Dict[str, Any],
    sampler_to_io: Dict[str, Any],
    scheduler_inference: IDDPM,
) -> None:
    """
    Main training loop.
    """

    running_loss = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        with tqdm(
            iterable=enumerate(iterable=dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            iteration_times = []
            for step, batch in pbar:
                start_time = time.time()
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")
                # calculate visual and text encoding
                with torch.no_grad():
                    model_args = dict()
                    # compress vid
                    x: torch.Tensor = vae.encode(x)  # [B, C, T, H/P, W/P]
                # generate masks
                mask: torch.Tensor = mask_generator.get_masks(x)
                model_args["x_mask"] = mask
                # video info and conditions
                for k, v in batch.items():
                    model_args[k] = push_to_device(item=v, device=device, dtype=dtype)
                # diffusion
                t: torch.Tensor = torch.randint(
                    low=0,
                    high=scheduler.num_timesteps,
                    size=(x.shape[0],),
                    device=device,
                )
                loss_dict = scheduler.training_losses(
                    model, x, t, model_args, mask=mask
                )
                # backward & update
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                # update EMA
                update_ema(ema, model.module, optimizer=optimizer)
                # log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1
                iteration_times.append(time.time() - start_time)
                
                # log to tensorboard
                if coordinator.is_master() and global_step % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix(
                        {"loss": avg_loss, "step": step, "global_step": global_step}
                    )
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)
                    weight_norm = calculate_weight_norm(model)
                    # TODO: log training reconstruction and trajectory.
                    if cfg.wandb:
                        wandb.log(
                            {
                                "avg_iteration_time": sum(iteration_times)
                                / len(iteration_times),
                                "iter": global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "acc_step": acc_step,
                                "lr": optimizer.param_groups[0]["lr"],
                                "weight_norm": weight_norm,
                            },
                            step=global_step,
                        )
                        iteration_times = []
                # save checkpoint
                if (
                    cfg.ckpt_every > 0
                    and global_step % cfg.ckpt_every == 0
                    and global_step != 0
                ):
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        cfg.batch_size,
                        coordinator,
                        exp_dir,
                        ema_shape_dict,
                        sampler=sampler_to_io,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

                    # log prompts for each checkpoints
                if global_step % cfg.eval_steps == 0:
                    write_sample(
                        model,
                        vae,
                        scheduler_inference,
                        cfg,
                        epoch,
                        exp_dir,
                        global_step,
                        dtype,
                        device,
                    )
                    log_sample(
                        coordinator.is_master(), cfg, epoch, exp_dir, global_step
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0


def main():

    # parse command-line args
    args = parse_configs(training=True)
    cfg: TrainingConfig = TrainingConfig(args)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg.to_dict(), exp_dir)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)

    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    writer: Optional[Any] = None
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        logger = create_logger(exp_dir)
        logger.info("Training configuration:")
        pprint(cfg.to_dict())
        logger.info(f"Experiment directory created at {exp_dir}")
        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            PROJECT = cfg.wandb_project_name
            wandb.init(
                project=PROJECT,
                entity=cfg.wandb_project_entity,
                name=exp_name,
                config=cfg.to_dict(),
            )

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset: VariableNBAClipsDataset = VariableNBAClipsDataset(
        data_path=cfg.dataset.data_path,
        num_frames=cfg.dataset.num_frames,
        frame_interval=cfg.dataset.frame_interval,
        image_size=cfg.dataset.image_size,
        transform_name=cfg.dataset.transform_name,
    )
    logger.info(f"Dataset contains {len(dataset)} samples.")
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    # TODO: use plugin's prepare dataloader
    dataloader = prepare_variable_dataloader(
        bucket_config=cfg.bucket_config,
        num_bucket_build_workers=cfg.num_bucket_build_workers,
        **dataloader_args,
    )

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model

    # the autoencoder, compresses videos by spatial and temp axis
    vae: VideoAutoencoderKL = build_module(cfg.vae.to_dict(), MODELS)
    input_size = (dataset.num_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(input_size)

    # construct the  diffusion transformer
    # we use: STDiT2-XL/2
    model: STDiT2 = build_module(
        module=cfg.model.to_dict(),
        builder=MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=CAPTION_CHANNELS,
        model_max_length=MODEL_MAX_LENGTH,
    )
    # get parameter counts
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    # model that tracks exponential moving avg of params
    ema: STDiT2 = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict: Dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler: IDDPM = build_module(cfg.scheduler.to_dict(), SCHEDULERS)
    scheduler_inference: IDDPM = build_module(
        cfg.scheduler_inference.to_dict(), SCHEDULERS
    )

    # 4.5. setup optimizer
    optimizer: HybridAdam = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )

    lr_scheduler: ConstantWarmupLR = ConstantWarmupLR(
        optimizer, factor=1, warmup_steps=cfg.warmup_steps, last_epoch=-1
    )

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()

    # TODO: mask ratios are never `None`
    mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")
    
    assert type(dataloader.batch_sampler) is VariableNBAClipsBatchSampler

    # TODO: we always use VariableVideoTextDataset
    num_steps_per_epoch = (
        dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    )

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    sampler_to_io = (
        dataloader.batch_sampler
        if cfg.dataset.type == "VariableVideoTextDataset"
        else None
    )

    # TODO: we never load from a checkpoint
    logger.info(
        f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch"
    )

    model_sharding(ema)

    # log prompts for pre-training ckpt
    first_global_step = start_epoch * num_steps_per_epoch + start_step
    write_sample(
        model,
        vae,
        scheduler_inference,
        cfg,
        start_epoch,
        exp_dir,
        first_global_step,
        dtype,
        device,
    )
    log_sample(coordinator.is_master(), cfg, start_epoch, exp_dir, first_global_step)
    logger.info("First global step done")

    # train the model
    train(
        cfg,
        coordinator,
        logger,
        vae,
        model,
        scheduler,
        start_epoch,
        dataloader,
        mask_generator,
        num_steps_per_epoch,
        device,
        dtype,
        booster,
        optimizer,
        lr_scheduler,
        ema,
        writer,
        exp_dir,
        ema_shape_dict,
        sampler_to_io,
        scheduler_inference,
    )


if __name__ == "__main__":
    main()
