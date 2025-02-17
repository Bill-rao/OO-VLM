#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
import torch.nn.functional as F
import torch.distributed
import tqdm
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from timm.utils import NativeScaler

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint_amp as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter, Fusion_TrainMeter, Fusion_ValMeter
from slowfast.utils.multigrid import MultigridSchedule

from slowfast.models.fusion import fusion_acc

logger = logging.get_logger(__name__)


def train_epoch(
        train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_scaler (scaler): scaler for loss.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels, indexes, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        # if cur_iter >= 40: # TODO 删除
        #     break

        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], str):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # 判断当前iter是否需要进行梯度累积 True则不更新梯度
        if cfg.TRAIN.GRADIENT_ACCUMULATE and (cur_iter + 1) % cfg.TRAIN.GRADIENT_ACCUMULATE == 0:
            is_need_gradient_accumulate = True
        else:
            is_need_gradient_accumulate = False
        # is_need_gradient_accumulate = False if (not cfg.TRAIN.GRADIENT_ACCUMULATE and (cur_iter + 1) % cfg.TRAIN.GRADIENT_ACCUMULATE != 0) else True
        # logger.info(f"is_need_gradient_accumulate: {is_need_gradient_accumulate} iter:{cur_iter}")
        # Update the learning rate.
        if not is_need_gradient_accumulate:  # 每个梯度累积的 minibatch 的训练参数保持不变
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast():
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, meta["boxes"])
            elif cfg.FUSION.ENABLE:
                # if not is_need_gradient_accumulate or not cfg.TRAIN.GRADIENT_ACCUMULATE:
                #     if cur_epoch > 0:
                #         alpha = cfg.FUSION.ALPHA
                #     else:
                #         alpha = cfg.FUSION.ALPHA * min(1., cur_iter / len(train_loader))

                # loss = model(inputs, indexes, meta, alpha)
                loss_vtc, loss_vtm, cls_result = model(inputs, indexes, meta, 0)

                cls_loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(label_smoothing=0.1)
                cls_loss = cls_loss_fun(cls_result, labels)

                loss = 0.2 * loss_vtc + 0.4 * loss_vtm + 0.4 * cls_loss
            else:
                if cfg.MODEL.MODEL_NAME.lower() == "coord":
                    preds = model(meta["box_categories"], meta["box_tensors"])
                    # logger.info(f"preds: {preds.shape}")
                else:
                    preds = model(inputs, indexes, meta)

                # Explicitly declare reduction to mean.
                loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

                # Compute the loss.
                loss = loss_fun(preds, labels)

            if cfg.TRAIN.GRADIENT_ACCUMULATE:
                loss = loss / cfg.TRAIN.GRADIENT_ACCUMULATE

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        # scaler => backward and step
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRADIENT,
                    parameters=model.parameters(), create_graph=is_second_order)
        if is_need_gradient_accumulate:
            # logger.info(f"iter:{cur_iter} skip loss")
            train_meter.iter_toc()  # measure allreduce for this meter
            train_meter.log_iter_stats(cur_epoch, cur_iter)
            train_meter.iter_tic()
            continue  # 梯度累计过程，无需进行记录

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )
        if cfg.FUSION.ENABLE:
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])  # 获取所有loss
                [loss_vtc] = du.all_reduce([loss_vtc])  # 获取所有loss
                [loss_vtm] = du.all_reduce([loss_vtm])  # 获取所有loss
                [cls_loss] = du.all_reduce([cls_loss])
            loss = loss.item()
            loss_vtc = loss_vtc.item()
            loss_vtm = loss_vtm.item()
            cls_loss = cls_loss.item()
            train_meter.update_stats(loss, loss_vtc, loss_vtm, cls_loss, lr, inputs[0].size(0) * max(cfg.NUM_GPUS, 1))
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/loss_vtc": loss_vtc,
                        "Train/loss_vtm": loss_vtm,
                        "Train/cls_loss": cls_loss,
                        "Train/lr": lr
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        loss_scaler (scaler): scaler for loss.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    if cfg.FUSION.ENABLE:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        device = torch.device("cuda")
        for cur_iter, (inputs, labels, indexes, meta) in enumerate(val_loader):
            # if cur_iter > 40:  # TODO:删除
            #     break
            # logger.info(f"val iter: {cur_iter}")
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if not isinstance(val[i], str):
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            val_meter.data_toc()

            with torch.cuda.amp.autocast():
                # rgb_embeds = model.rgb_ln(model.rgb_encoder(inputs))
                rgb_embeds = model.rgb_ln(inputs)
                rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long, device=device)
                # rgb_feat = F.normalize(self.rgb_proj(rgb_embeds[:, 0, :]), dim=-1)

                coord_embeds = model.coord_ln(model.coord_encoder(meta["box_categories"], meta["box_tensors"]))
                coord_embeds = model.rgb_coord_uniform_proj(coord_embeds)
                coord_atts = torch.ones(coord_embeds.size()[:-1], dtype=torch.long, device=device)
                # coord_feat = F.normalize(self.coord_proj(coord_embeds[:, 0, :]), dim=-1)

                # Query-Vision
                query_tokens_cls = model.query_tokens.expand(rgb_embeds.shape[0], -1, -1)

                # encoder_hidden_states = [coord_embeds, rgb_embeds]
                # encoder_attention_mask = [coord_atts, rgb_atts]
                encoder_hidden_states = torch.cat([coord_embeds, rgb_embeds], dim=1)
                encoder_attention_mask = torch.cat([coord_atts, rgb_atts], dim=1)
                encoder_hidden_states = model.coord_rgb_add_embedding(encoder_hidden_states, coord_embeds.shape[1], rgb_embeds.shape[1])

                # query_output_cls = model.language_encoder(
                #     query_embeds=query_tokens_cls,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask,
                #     # use_cache=True, # TODO 不使用cache
                #     return_dict=True,
                # )
                # # query_feats = F.normalize(query_output.last_hidden_state, dim=-1)  # [32, 32, 768]
                # query_feats_cls = query_output_cls.last_hidden_state
                # cls_feat = query_feats_cls.mean(dim=1)  # [32, 768]
                # cls_result = model.classifier(cls_feat)  # [32, 174]
                cls_result = model.forward_classification(encoder_hidden_states, encoder_attention_mask, query_tokens_cls)

            num_topks_correct = metrics.topks_correct(cls_result, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / cls_result.size(0)) * 100.0 for x in num_topks_correct
            ]

            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()  # measure allreduce for this meter
            val_meter.update_stats(top1_err=top1_err, top5_err=top5_err, mb_size=inputs[0].size(0) * max(cfg.NUM_GPUS, 1))
            val_meter.update_predictions(cls_result, labels)
            val_meter.log_iter_stats(cur_epoch, cur_iter)
            val_meter.iter_tic()
        flag = val_meter.log_epoch_stats(cur_epoch)

        # texts = list(val_loader.dataset.bbox_dataset.all_text2id.keys())
        # num_text = len(texts)  # 174
        #
        # assert num_text == cfg.MODEL.NUM_CLASSES, "num of class does not match"
        #
        # text_ids = []
        # text_embeds = []
        # text_atts = []
        #
        # text_bs = cfg.TRAIN.BATCH_SIZE  # 使用测试设置的batch size
        # for i in range(0, num_text, text_bs):  # 根据 text_bs 取样
        #     text = texts[i: min(num_text, i + text_bs)]  # 拿到 第i个 text_bs
        #     text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=cfg.FUSION.MAX_WORDS, return_tensors="pt").to(device)
        #     text_output = model.language_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, return_dict=True)
        #     text_embed = F.normalize(model.language_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        #
        #     text_embeds.append(text_embed)
        #     text_ids.append(text_input.input_ids)
        #     text_atts.append(text_input.attention_mask)
        #
        # text_embeds = torch.cat(text_embeds, dim=0)  # [174, 512]
        # text_ids = torch.cat(text_ids, dim=0)  # [174, 30]
        # text_atts = torch.cat(text_atts, dim=0)  # [174, 30]
        # # text_ids[:, 0] = model.tokenizer.enc_token_id
        #
        # # init 分数矩阵
        # scores_q2t = torch.full((len(val_loader.dataset.bbox_dataset.all_vision), len(texts)), -100.0).cuda()  # [22659, 174]
        # scores_vtm = torch.full((len(val_loader.dataset.bbox_dataset.all_vision), len(texts)), -100.0).cuda()  # [22659, 174]
        #
        # for cur_iter, (inputs, labels, indexes, meta) in enumerate(val_loader):
        #     # if cur_iter > 40:  # TODO:删除
        #     #     break
        #     # logger.info(f"val iter: {cur_iter}")
        #     if cfg.NUM_GPUS:
        #         # Transferthe data to the current GPU device.
        #         if isinstance(inputs, (list,)):
        #             for i in range(len(inputs)):
        #                 inputs[i] = inputs[i].cuda(non_blocking=True)
        #         else:
        #             inputs = inputs.cuda(non_blocking=True)
        #         labels = labels.cuda()
        #         for key, val in meta.items():
        #             if isinstance(val, (list,)):
        #                 for i in range(len(val)):
        #                     if not isinstance(val[i], str):
        #                         val[i] = val[i].cuda(non_blocking=True)
        #             else:
        #                 meta[key] = val.cuda(non_blocking=True)
        #     val_meter.data_toc()
        #
        #     rgb_feat = model.rgb_ln(model.rgb_encoder(inputs))  # [B*NUM_SAMPLE 1+T 768]
        #     rgb_atts = torch.ones(rgb_feat.size()[:-1], dtype=torch.long).cuda()  # [B*NUM_SAMPLE 1+T]
        #     # rgb_feat = rgb_feat.unsqueeze(1)  # [B*NUM_SAMPLE 1 D] 维度增加 TODO：采用cls token的全局特征信息
        #     # rgb_embed = F.normalize(model.rgb_proj(rgb_feat[:, 0, :]), dim=-1)  # [B*NUM_SAMPLE, 512]
        #
        #     coord_feat = model.coord_encoder(meta["box_categories"], meta["box_tensors"])  #
        #     coord_feat = model.rgb_coord_uniform_proj(coord_feat)  # [32, 33, 768]
        #     coord_atts = torch.ones(coord_feat.size()[:-1], dtype=torch.long).cuda()  #
        #     # coord_embed = F.normalize(model.coord_proj(coord_feat[:, 0, :]), dim=-1)  # [32, 512]
        #
        #     query_tokens = model.query_tokens.expand(rgb_feat.shape[0], -1, -1)
        #
        #     encoder_hidden_states = [coord_feat, rgb_feat]
        #     encoder_attention_mask = [coord_atts, rgb_atts]
        #     query_output = model.language_encoder(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         # use_cache=True, # TODO 不使用cache
        #         return_dict=True,
        #     )
        #     query_feats = F.normalize(model.query_proj(query_output.last_hidden_state), dim=-1)
        #
        #     sim_q2t = torch.matmul(query_feats.unsqueeze(1), text_embeds.unsqueeze(-1)).squeeze()
        #     sim_q2t_max, _ = sim_q2t.max(-1)
        #     sim_q2t_max = sim_q2t_max / model.temp  # [48, 174]
        #
        #     for index in range(sim_q2t_max.shape[0]):
        #         # query_feats_repeat = query_feats[index].repeat(text_embeds.shape[0], 1, 1).cuda()  # [174, 32, 512]
        #
        #         query_tokens = model.query_tokens.expand(text_embeds.shape[0], -1, -1).cuda()
        #         query_att_repeat = torch.ones(query_tokens.size()[:-1], dtype=torch.long).cuda()  # [174, 32]
        #         attention_mask = torch.cat([query_att_repeat, text_atts], dim=1)  # [174, 62]
        #
        #         coord_feat_repeat = coord_feat[index].repeat(text_embeds.shape[0], 1, 1).to(device)  # [174, 33, 768]
        #         coord_att_repeat = torch.ones(coord_feat_repeat.size()[:-1], dtype=torch.long).to(device)  # [174, 33]
        #
        #         rgb_feat_repeat = rgb_feat[index].repeat(text_embeds.shape[0], 1, 1).to(device)  # [48, 9, 768]
        #         rgb_att_repeat = torch.ones(rgb_feat_repeat.size()[:-1], dtype=torch.long).to(device)  # [174, 9]
        #
        #         encoder_hidden_states = [coord_feat_repeat, rgb_feat_repeat]
        #         encoder_attention_mask = [coord_att_repeat, rgb_att_repeat]
        #
        #         output = model.language_encoder(text_ids,
        #                                         query_embeds=query_tokens,
        #                                         attention_mask=attention_mask,
        #                                         encoder_hidden_states=encoder_hidden_states,
        #                                         encoder_attention_mask=encoder_attention_mask,
        #                                         return_dict=True,
        #                                         )
        #
        #         score = model.vtm_head(output.last_hidden_state[:, :query_tokens_vtm.size(1), :])[:, 1]
        #
        #         # scores_c2t[cur_iter * val_loader.batch_size + index] = sims_c2t[index] + score  # [174]
        #         # scores_r2t[cur_iter * val_loader.batch_size + index] = sims_r2t[index] + score  # [174]
        #         # scores_mix[cur_iter * val_loader.batch_size + index] = sims_c2t[index] + sims_r2t[index] + score  # [174]
        #
        #         # print(f"cur_iter: {cur_iter}  test_loader.batch_size:{val_loader.batch_size}  index:{index}  text_embeds.shape[0]:{text_embeds.shape[0]}")
        #
        #         # sim_r2t = sims_r2t[index]  # [174]
        #         # sim_c2t = sims_c2t[index]  # [174]
        #         #
        #         # topk_sim_r2t, topk_r2t_idx = sim_r2t.topk(k=k_test, dim=0)  # [k_test], [k_test]
        #         # topk_sim_c2t, topk_c2t_idx = sim_c2t.topk(k=k_test, dim=0)  # [k_test], [k_test]
        #         #
        #         # encoder_output = coord_feat[index].repeat(k_test, 1, 1).to(device)
        #         # encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        #         # output_coord = model.language_encoder(text_ids[topk_c2t_idx],
        #         #                                       attention_mask=text_atts[topk_c2t_idx],
        #         #                                       encoder_hidden_states=encoder_output,
        #         #                                       encoder_attention_mask=encoder_att,
        #         #                                       return_dict=True,
        #         #                                       mode='multimodal_pos'
        #         #                                       )
        #         # score_coord = model.vtm_head_coord(output_coord.last_hidden_state[:, 0, :])[:, 1]  # [174]
        #         # scores_c2t[cur_iter * val_loader.batch_size + index, topk_c2t_idx] = score_coord + topk_sim_c2t
        #         #
        #         # encoder_output = rgb_feat[index].repeat(k_test, 1, 1).to(device)
        #         # encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        #         # output_rgb = model.language_encoder(text_ids[topk_r2t_idx],
        #         #                                     # text_ids[topk_r2t_idx],
        #         #                                     # attention_mask=text_atts[topk_r2t_idx],
        #         #                                     attention_mask=text_atts[topk_r2t_idx],
        #         #                                     encoder_hidden_states=encoder_output,
        #         #                                     encoder_attention_mask=encoder_att,
        #         #                                     return_dict=True,
        #         #                                     mode='multimodal_rgb'
        #         #                                     )
        #         # score_rgb = model.vtm_head_rgb(output_rgb.last_hidden_state[:, 0, :])[:, 1]  # [174]
        #         # scores_r2t[cur_iter * val_loader.batch_size + index, topk_r2t_idx] = score_rgb + topk_sim_r2t
        #
        #     val_meter.iter_toc()  # measure allreduce for this meter
        #     val_meter.log_iter_stats(cur_iter, len(val_loader), "cal_sims")
        #     val_meter.iter_tic()
        #
        # # scores_mix = scores_c2t + scores_r2t  # [67977, 174]
        #
        # r1_r2t, r5_r2t = fusion_acc(scores_r2t.cpu().numpy(), val_loader.dataset.bbox_dataset.vision2text)
        # r1_c2t, r5_c2t = fusion_acc(scores_c2t.cpu().numpy(), val_loader.dataset.bbox_dataset.vision2text)
        # r1_mix, r5_mix = fusion_acc(scores_mix.cpu().numpy(), val_loader.dataset.bbox_dataset.vision2text)
        #
        # # Val 使用了sampler 多卡推理，因此最后验证结果需要重新汇总计算
        # if cfg.NUM_GPUS > 1:
        #     r1_c2t, r1_r2t, r1_mix = torch.tensor(r1_c2t).cuda(), torch.tensor(r1_r2t).cuda(), torch.tensor(r1_mix).cuda()
        #     r5_c2t, r5_r2t, r5_mix = torch.tensor(r5_c2t).cuda(), torch.tensor(r5_r2t).cuda(), torch.tensor(r5_mix).cuda()
        #     r1_c2t, r1_r2t, r1_mix, r5_c2t, r5_r2t, r5_mix = du.all_reduce([r1_c2t, r1_r2t, r1_mix, r5_c2t, r5_r2t, r5_mix], average=True)
        #     r1_c2t, r1_r2t, r1_mix, r5_c2t, r5_r2t, r5_mix = r1_c2t.item(), r1_r2t.item(), r1_mix.item(), r5_c2t.item(), r5_r2t.item(), r5_mix.item()
        #
        # val_meter.update_predictions(r1_c2t, r5_c2t, r1_r2t, r5_r2t, r1_mix, r5_mix)
        # flag = val_meter.log_final_stats(cur_epoch)
        # val_meter.reset()
        #
        # # flag = True  # TODO: 这里需要重写 flag=True 意味着将会save best.pth

    else:
        for cur_iter, (inputs, labels, indexes, meta) in enumerate(val_loader):
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if not isinstance(val[i], str):
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            val_meter.data_toc()

            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
                ori_boxes = meta["ori_boxes"]
                metadata = meta["metadata"]

                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    ori_boxes = ori_boxes.cpu()
                    metadata = metadata.cpu()

                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(preds, ori_boxes, metadata)
            else:
                if cfg.MODEL.MODEL_NAME.lower() == "coord":
                    preds = model(meta["box_categories"], meta["box_tensors"])
                else:
                    preds = model(inputs)

                if cfg.DATA.MULTI_LABEL:
                    if cfg.NUM_GPUS > 1:
                        preds, labels = du.all_gather([preds, labels])
                else:
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                    # Combine the errors across the GPUs.
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        top1_err,
                        top5_err,
                        inputs[0].size(0)
                        * max(
                            cfg.NUM_GPUS, 1
                        ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                    )
                    # write to tensorboard format if available.
                    if writer is not None:
                        writer.add_scalars(
                            {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                            global_step=len(val_loader) * cur_epoch + cur_iter,
                        )

                val_meter.update_predictions(preds, labels)

            val_meter.log_iter_stats(cur_epoch, cur_iter)
            val_meter.iter_tic()

        # Log epoch stats.
        flag = val_meter.log_epoch_stats(cur_epoch)

        # write to tensorboard format if available.
        if writer is not None:
            if cfg.DETECTION.ENABLE:
                writer.add_scalars(
                    {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
                )
            else:
                all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
                all_labels = [
                    label.clone().detach() for label in val_meter.all_labels
                ]
                if cfg.NUM_GPUS:
                    all_preds = [pred.cpu() for pred in all_preds]
                    all_labels = [label.cpu() for label in all_labels]
                writer.plot_eval(
                    preds=all_preds, labels=all_labels, global_step=cur_epoch
                )

        val_meter.reset()
    return flag


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Loss scaler
    loss_scaler = NativeScaler()

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    if cfg.FUSION.ENABLE:
        train_meter = Fusion_TrainMeter(len(train_loader), cfg)
        val_meter = Fusion_ValMeter(len(val_loader), cfg)
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        loss_scaler,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
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

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Loss scaler
    loss_scaler = NativeScaler()

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, loss_scaler)
    # # additional 设置学习率
    # if cfg.TRAIN.IS_RESET_EPOCH:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = cfg.SOLVER.BASE_LR

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    elif cfg.FUSION.ENABLE:
        train_meter = Fusion_TrainMeter(len(train_loader), cfg)
        val_meter = Fusion_ValMeter(len(val_loader), cfg)
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    loss_scaler,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # data_recorder epoch
        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()
        train_loader.dataset.rgb_dataset.update_epoch_data(cur_epoch, dataset_or_feature='feature')
        # val_loader.dataset.rgb_dataset.update_epoch_data(cur_epoch, dataset_or_feature='feature')  # TODO 测试val阶段是否需要重新打乱样本顺序
        if cfg.NUM_GPUS > 1:
            torch.distributed.barrier()

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer
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
            f"{epoch_timer.last_epoch_time() / len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time() / len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )
        # is_eval_epoch = True

        # Compute precise BN stats.
        if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch or cfg.TRAIN.SAVE_LATEST:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            flag = eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer)
            if flag:
                cu.save_best_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)

    if writer is not None:
        writer.close()
