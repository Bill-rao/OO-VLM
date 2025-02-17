#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr
import pprint
from einops import rearrange, repeat

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.datasets.mixup import MixUp
import slowfast.utils.metrics as metrics
from tools.data_record.record_meters import TestMeter, GetFeatureMeter, ValMeter, EpochTimer

from tools.data_record.data_record import DataRecorder, construct_loader

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, data_recorder: DataRecorder, writer=None):
    # Enable eval mode.
    # model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, video_index, index, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            index = index.cuda()
            video_index = video_index.cuda()

        test_meter.data_toc()

        # ADD_SOFTMAX
        if cfg.TEST.ADD_SOFTMAX:
            preds = inputs.softmax(-1)
        else:
            preds = inputs

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, index, labels, video_index = du.all_gather(
                [preds, index, labels, video_index]
            )

        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_index = video_index.cpu()
            index = index.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_index.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter


def test_feature(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # data recorder
    data_recoder = DataRecorder(cfg.DATARECORDER.OUTPUT_FOLDER_PATH, cfg.DATARECORDER.DATASPLIT)

    # Create video testing loaders.
    data_recoder.init_dataset(cfg, dataset_or_feature='feature')
    test_feature_loader = construct_loader(data_recoder, batch_size=cfg.TEST.BATCH_SIZE,
                                           sampler=DistributedSampler(data_recoder, shuffle=False) if cfg.NUM_GPUS > 1 else None,
                                           shuffle=False,
                                           num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                           pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                                           drop_last=False)

    logger.info("Testing model for {} iterations".format(len(test_feature_loader)))
    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")

    # assert (
    #         test_feature_loader.dataset.num_samples
    #         % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
    #         == 0
    # )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        cfg,
        len(test_feature_loader.dataset) // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_feature_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    test_meter = perform_test(test_feature_loader, None, test_meter, cfg, data_recoder, None)


@torch.no_grad()
def get_feature_epoch(
        get_feature_loader, model, optimizer, loss_scaler, get_feature_meter, cur_epoch, cfg, data_recorder: DataRecorder, writer=None
):
    # Just to get the feature vector, eval model
    # model.train()
    model.eval()
    get_feature_meter.iter_tic()
    data_size = len(get_feature_loader)

    # if cfg.MIXUP.ENABLE:
    #     mixup_fn = MixUp(
    #         mixup_alpha=cfg.MIXUP.ALPHA,
    #         cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
    #         mix_prob=cfg.MIXUP.PROB,
    #         switch_prob=cfg.MIXUP.SWITCH_PROB,
    #         label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
    #         num_classes=cfg.MODEL.NUM_CLASSES,
    #     )

    for cur_iter, (inputs, labels, index, meta) in enumerate(get_feature_loader):
        # if cur_iter >= 12:  # 删除
        #     break

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            if cfg.MODEL.ARCH in ['avion']:
                inputs = inputs.permute(0, 4, 1, 2, 3)
                inputs = get_feature_loader.dataset.transform_gpu(inputs)

            labels = labels.cuda()
            index = index.cuda()

            video_index = meta['video_index']
            if len(video_index.shape) > 1:
                video_index = video_index.flatten()
            video_index = video_index.cuda(non_blocking=True)

            if cfg.MODEL.ARCH in ['omnimae']:
                inputs = rearrange(inputs, "B A C T H W -> (B A) C T H W").cuda()
                labels = rearrange(labels, "B A -> (B A)").cuda()
                index = rearrange(index, "B A -> (B A)").cuda()

            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        else:
            video_index = meta['video_index']

        get_feature_meter.data_toc()

        with torch.cuda.amp.autocast():
            output_feature = model(inputs)

            # wait all GPUs
            if cfg.NUM_GPUS > 1:
                dist.barrier()

            if cfg.NUM_GPUS > 1:
                output_feature, index, labels, video_index = du.all_gather(
                    [output_feature, index, labels, video_index]
                )

            if cfg.NUM_GPUS:
                output_feature = output_feature.cpu()
                index = index.cpu()
                labels = labels.cpu()
                video_index = video_index.cpu()

            # data_recoder
            if du.is_master_proc():
                data_recorder.update_feature_data(output_feature, sample_index=index, label=labels, video_index=video_index, epoch=cur_epoch)

            get_feature_meter.iter_toc()  # measure allreduce for this meter
            get_feature_meter.log_iter_stats(cur_epoch, cur_iter)
            get_feature_meter.iter_tic()

    if du.is_master_proc():
        data_recorder.save_feature_data()

    # Log epoch stats.
    get_feature_meter.log_epoch_stats(cur_epoch)
    get_feature_meter.reset()


def test_get_feature(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Get feature with config:")
    logger.info(pprint.pformat(cfg))

    # data recorder
    if du.is_master_proc():
        data_recoder = DataRecorder(cfg.DATARECORDER.OUTPUT_FOLDER_PATH, cfg.DATARECORDER.DATASPLIT)
    else:
        data_recoder = None

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Load a checkpoint to resume training if applicable.
    # start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, loss_scaler)
    start_epoch = 0

    if cfg.MODEL.ARCH not in ['avion', 'omnimae']:
        load_checkpoint(path_to_checkpoint=cfg.TEST.CHECKPOINT_FILE_PATH,
                        model=model,
                        loss_scaler=None,
                        data_parallel=cfg.NUM_GPUS > 1,
                        inflation=False)

    # Create the video train and val loaders.
    get_feature_loader = loader.construct_loader(cfg, cfg.DATARECORDER.DATASPLIT)
    # val_loader = loader.construct_loader(cfg, "val")
    # precise_bn_loader = (
    #     loader.construct_loader(cfg, "train", is_precise_bn=True)
    #     if cfg.BN.USE_PRECISE_STATS
    #     else None
    # )

    # Create meters.
    get_feature_meter = GetFeatureMeter(len(get_feature_loader), cfg)
    # val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(get_feature_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        get_feature_epoch(
            get_feature_loader, model, None, None, get_feature_meter, cur_epoch, cfg, data_recoder, None
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time() / len(get_feature_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time() / len(get_feature_loader):.2f}s in average."
        )

