#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import torch.utils.data
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, Fusion_TestMeter
from slowfast.models.fusion import fusion_acc

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader: torch.utils.data.DataLoader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter : testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    if cfg.FUSION.ENABLE:
        # all_labels = []
        # all_indexes = []
        device = torch.device("cuda")
        for cur_iter, (inputs, labels, sample_indexes, meta) in enumerate(test_loader):
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                sample_indexes = sample_indexes.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if not isinstance(val[i], str):
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

                video_index = meta['video_index']
            # all_labels.append(labels)
            # all_indexes.append(sample_indexes)

            test_meter.data_toc()

            with torch.cuda.amp.autocast():
                # rgb_embeds = model.rgb_ln(model.rgb_encoder(inputs))
                rgb_embeds = model.rgb_ln(inputs)
                rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long, device=device)
                # rgb_feat = F.normalize(self.rgb_proj(rgb_embeds[:, 0, :]), dim=-1)

                coord_embeds = model.coord_ln(model.coord_encoder(meta["box_categories"], meta["box_tensors"]))
                coord_embeds = model.rgb_coord_uniform_proj(coord_embeds)
                coord_atts = torch.ones(coord_embeds.size()[:-1], dtype=torch.long).cuda()
                # coord_feat = F.normalize(self.coord_proj(coord_embeds[:, 0, :]), dim=-1)

                # Query-Vision
                query_tokens_cls = model.query_tokens.expand(rgb_embeds.shape[0], -1, -1)

                # encoder_hidden_states = [coord_embeds, rgb_embeds]
                # encoder_attention_mask = [coord_atts, rgb_atts]
                encoder_hidden_states = torch.cat([coord_embeds, rgb_embeds], dim=1)
                encoder_attention_mask = torch.cat([coord_atts, rgb_atts], dim=1)
                encoder_hidden_states = model.coord_rgb_add_embedding(encoder_hidden_states, coord_embeds.shape[1], rgb_embeds.shape[1])

                # query_output = model.language_encoder(
                #     query_embeds=query_tokens_cls,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask,
                #     # use_cache=True, # TODO 不使用cache
                #     return_dict=True,
                # )
                # query_feats = F.normalize(query_output.last_hidden_state, dim=-1)  # [32, 32, 768]
                # cls_feat = query_feats.mean(dim=1)  # [32, 768]
                # cls_result = model.classifier(cls_feat)  # [32, 174]
                cls_result = model.forward_classification(encoder_hidden_states, encoder_attention_mask, query_tokens_cls)

            if cfg.TEST.ADD_SOFTMAX:
                cls_result = cls_result.softmax(dim=-1)

            if cfg.NUM_GPUS:
                cls_result = cls_result.cpu()
                labels = labels.cpu()
                sample_indexes = sample_indexes.cpu()

            test_meter.iter_toc()
            # test_meter.update_stats(cls_result.detach(),
            #                         labels.detach(),
            #                         sample_indexes.detach(),
            #                         )
            test_meter.update_stats(
                cls_result.detach(), labels.detach(), video_index.detach(),
            )
            test_meter.log_iter_stats(cur_iter)
            test_meter.iter_tic()
        test_meter.finalize_metrics()

        # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #     model = model.module
        # device = torch.device("cuda")
        #
        # texts = list(test_loader.dataset.bbox_dataset.all_text2id.keys())
        # num_text = len(texts)  # 174
        #
        # assert num_text == cfg.MODEL.NUM_CLASSES, "num of class does not match"
        #
        # text_ids = []
        # text_embeds = []
        # text_atts = []
        # text_label_ids = []
        #
        # text_bs = cfg.TEST.BATCH_SIZE  # 使用测试设置的batch size
        # for i in range(0, num_text, text_bs):  # 根据 text_bs 取样
        #     text = texts[i: min(num_text, i + text_bs)]  # 拿到 第i个 text_bs
        #     for t in text:
        #         text_label_id = test_loader.dataset.get_label_id(t)
        #         text_label_ids.append(text_label_id)
        #
        #     text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        #     text_output = model.language_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        #     text_embed = F.normalize(model.language_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        #
        #     text_embeds.append(text_embed)
        #     text_ids.append(text_input.input_ids)
        #     text_atts.append(text_input.attention_mask)
        #
        # text_embeds = torch.cat(text_embeds, dim=0)  # [174, 512]
        # text_ids = torch.cat(text_ids, dim=0)
        # text_atts = torch.cat(text_atts, dim=0)
        # text_ids[:, 0] = model.tokenizer.enc_token_id
        #
        # # other data
        # all_labels = []
        # all_indexes = []
        #
        # # init 分数矩阵
        # scores_r2t = torch.full(
        #     (len(test_loader.dataset.bbox_dataset.all_vision) * (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        #      len(texts)), -100.0).cuda()  # [22659, 174]
        # scores_c2t = torch.full(
        #     (len(test_loader.dataset.bbox_dataset.all_vision) * (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        #      len(texts)), -100.0).cuda()  # [22659, 174]
        # scores_mix = torch.full(
        #     (len(test_loader.dataset.bbox_dataset.all_vision) * (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        #      len(texts)), -100.0).cuda()  # [22659, 174]
        # for cur_iter, (inputs, labels, sample_indexes, meta) in enumerate(test_loader):
        #     # if cur_iter >= 200:  # TODO:删除
        #     #     break
        #     if cfg.NUM_GPUS:
        #         # Transferthe data to the current GPU device.
        #         if isinstance(inputs, (list,)):
        #             for i in range(len(inputs)):
        #                 inputs[i] = inputs[i].cuda(non_blocking=True)
        #         else:
        #             inputs = inputs.cuda(non_blocking=True)
        #         labels = labels.cuda()
        #         sample_indexes = sample_indexes.cuda()
        #         for key, val in meta.items():
        #             if isinstance(val, (list,)):
        #                 for i in range(len(val)):
        #                     if not isinstance(val[i], str):
        #                         val[i] = val[i].cuda(non_blocking=True)
        #             else:
        #                 meta[key] = val.cuda(non_blocking=True)
        #     all_labels.append(labels)
        #     all_indexes.append(sample_indexes)
        #
        #     test_meter.data_toc()
        #
        #     rgb_feat = model.rgb_encoder(inputs)  # [B*NUM_SAMPLE 1+T 768]
        #     # rgb_feat = rgb_feat.unsqueeze(1)  # [B*NUM_SAMPLE 1 D] 维度增加 TODO：采用cls token的全局特征信息
        #     rgb_embed = F.normalize(model.rgb_proj(rgb_feat[:, 0, :]), dim=-1)  # [B*NUM_SAMPLE, 512]
        #
        #     coord_feat = model.coord_encoder(meta["box_categories"], meta["box_tensors"])  #
        #     coord_feat = model.rgb_coord_uniform_proj(coord_feat)  # [32, 33, 768]
        #     # coord_atts = torch.ones(coord_embeds.size()[:-1], dtype=torch.long).cuda()  #
        #     coord_embed = F.normalize(model.coord_proj(coord_feat[:, 0, :]), dim=-1)  # [32, 512]
        #
        #     # 计算相似度矩阵
        #     sims_r2t = rgb_embed @ text_embeds.t()  # [32, 174]
        #     sims_c2t = coord_embed @ text_embeds.t()  # [32, 174]
        #
        #     assert sims_r2t.shape == sims_c2t.shape
        #     # k_test = sims_r2t.shape[1]
        #     for index in range(sims_r2t.shape[0]):
        #         coord_feat_repeat = coord_feat[index].repeat(text_embeds.shape[0], 1, 1).to(device)
        #         coord_att_repeat = torch.ones(coord_feat_repeat.size()[:-1], dtype=torch.long).to(device)
        #
        #         rgb_feat_repeat = rgb_feat[index].repeat(text_embeds.shape[0], 1, 1).to(device)
        #         rgb_att_repeat = torch.ones(rgb_feat_repeat.size()[:-1], dtype=torch.long).to(device)
        #
        #         encoder_hidden_states = [coord_feat_repeat, rgb_feat_repeat]
        #         encoder_attention_mask = [coord_att_repeat, rgb_att_repeat]
        #
        #         output = model.language_encoder(text_ids,
        #                                         attention_mask=text_atts,
        #                                         encoder_hidden_states=encoder_hidden_states,
        #                                         encoder_attention_mask=encoder_attention_mask,
        #                                         return_dict=True,
        #                                         mode='multimodal'
        #                                         )
        #         score = model.vtm_head(output.last_hidden_state[:, 0, :])[:, 1]
        #
        #         scores_c2t[cur_iter * test_loader.batch_size + index] = sims_c2t[index] + score  # [174]
        #         scores_r2t[cur_iter * test_loader.batch_size + index] = sims_r2t[index] + score  # [174]
        #         scores_mix[cur_iter * test_loader.batch_size + index] = sims_c2t[index] + sims_r2t[index] + score  # [174]
        #
        #     test_meter.iter_toc()  # measure allreduce for this meter
        #     test_meter.log_iter_stats(cur_iter, len(test_loader), "cal_sims")
        #     test_meter.iter_tic()
        #
        # all_labels = torch.cat(all_labels, dim=0)
        # all_indexes = torch.cat(all_indexes, dim=0)
        #
        # # scores_mix = scores_c2t + scores_r2t  # [67977, 174]
        #
        # if cfg.TEST.ADD_SOFTMAX:
        #     scores_r2t = scores_r2t.softmax(dim=-1)
        #     scores_c2t = scores_c2t.softmax(dim=-1)
        #     scores_mix = scores_mix.softmax(dim=-1)
        #
        # # BLIP 算法
        # r1_r2t, r5_r2t = fusion_acc(scores_r2t.cpu().numpy(),
        #                             test_loader.dataset.bbox_dataset.vision2text,
        #                             (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS))
        # r1_c2t, r5_c2t = fusion_acc(scores_c2t.cpu().numpy(),
        #                             test_loader.dataset.bbox_dataset.vision2text,
        #                             (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS))
        # r1_mix, r5_mix = fusion_acc(scores_mix.cpu().numpy(),
        #                             test_loader.dataset.bbox_dataset.vision2text,
        #                             (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS))
        #
        # # 重新排列 最后维度的排列顺序
        # scores_r2t_sorted = torch.zeros_like(scores_r2t, dtype=scores_r2t.dtype, device=scores_r2t.device)
        # scores_c2t_sorted = torch.zeros_like(scores_c2t, dtype=scores_c2t.dtype, device=scores_c2t.device)
        # text_sort_index = test_loader.dataset.bbox_dataset.all_text2label_id  # 根据 label 进行排序
        # assert len(text_sort_index) == scores_r2t_sorted.shape[1]
        # for i, idx in enumerate(text_sort_index):
        #     scores_r2t_sorted[:, idx] = scores_r2t[:, i]
        #     scores_c2t_sorted[:, idx] = scores_c2t[:, i]
        #
        # rgb_preds = scores_r2t_sorted  # [67977, 174]
        # coord_preds = scores_c2t_sorted  # [67977, 174]
        # if cfg.TEST.ADD_SOFTMAX:
        #     rgb_preds = rgb_preds.softmax(dim=-1)
        #     coord_preds = coord_preds.softmax(dim=-1)
        #
        # if cfg.NUM_GPUS:
        #     rgb_preds = rgb_preds.cpu()
        #     coord_preds = coord_preds.cpu()
        #     all_indexes = all_indexes.cpu()  # [168]
        #     all_labels = all_labels.cpu()  # [168]
        #
        # test_meter.update_stats(
        #     rgb_preds.detach(), coord_preds.detach(), all_labels.detach(), all_indexes.detach()
        # )
        # test_meter.finalize_metrics()
        #
        # logger.info(f"r1_r2t: {r1_r2t}   r1_c2t: {r1_c2t}   r1_mix: {r1_mix}")
        # logger.info(f"r5_r2t: {r5_r2t}   r5_c2t: {r5_c2t}   r5_mix: {r5_mix}")
        #
        # print("Pause")
        # while True: pass

        #############################################################
        #############################################################
        #############################################################
        #############################################################
        #############################################################

    else:
        for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)

                # Transfer the data to the current GPU device.
                labels = labels.cuda()
                video_idx = video_idx.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            test_meter.data_toc()

            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
                ori_boxes = meta["ori_boxes"]
                metadata = meta["metadata"]

                preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
                ori_boxes = (
                    ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
                )
                metadata = (
                    metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
                )

                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(preds, ori_boxes, metadata)
                test_meter.log_iter_stats(None, cur_iter)
            else:
                # Perform the forward pass.
                if cfg.TEST.ADD_SOFTMAX:
                    preds = model(inputs).softmax(-1)
                else:
                    preds = model(inputs)

                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, video_idx = du.all_gather(
                        [preds, labels, video_idx]
                    )
                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    labels = labels.cpu()
                    video_idx = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach(), labels.detach(), video_idx.detach()
                )
                test_meter.log_iter_stats(cur_iter)

            test_meter.iter_tic()

        # Log epoch stats and print the final testing results.
        if not cfg.DETECTION.ENABLE:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels
            if cfg.NUM_GPUS:
                all_preds = all_preds.cpu()
                all_labels = all_labels.cpu()
            if writer is not None:
                writer.plot_eval(preds=all_preds, labels=all_labels)

            if cfg.TEST.SAVE_RESULTS_PATH != "":
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

                if du.is_root_proc():
                    with g_pathmgr.open(save_path, "wb") as f:
                        pickle.dump([all_preds, all_labels], f)

                logger.info(
                    "Successfully saved prediction results to {}".format(save_path)
                )

        test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
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

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc(num_gpus=cfg.NUM_GPUS * cfg.NUM_SHARDS) and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    if cfg.FUSION.ENABLE:
        assert len(test_loader.dataset) % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0
        test_meter = Fusion_TestMeter(
            cfg,
            len(test_loader.dataset.rgb_dataset) // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            ensemble_method=cfg.DATA.ENSEMBLE_METHOD
        )
    else:
        assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),  # 样本数量
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,  # clip数量
            cfg.MODEL.NUM_CLASSES,  # 类别数量
            len(test_loader),  # iter 数量
            cfg.DATA.ENSEMBLE_METHOD,  # 计算方式 sum or max
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()

    file_name = f'{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_CROP_SIZE}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}.pkl'
    with g_pathmgr.open(os.path.join(
            cfg.OUTPUT_DIR, file_name),
            'wb'
    ) as f:
        result = {
            'video_preds': test_meter.video_preds,
            'video_labels': test_meter.video_labels,
            'top1': test_meter.stats["top1_acc"]
        }
        pickle.dump(result, f)
