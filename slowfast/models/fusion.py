import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import numpy as np
from collections import OrderedDict
from einops import rearrange

from .build import MODEL_REGISTRY

from .coord_model import VideoModelCoord
from transformers import BertTokenizer

# from .bert_med import BertConfig, BertModel
from .bert_med_query import BertConfig, BertModel

from slowfast.models.uniformerv2 import Uniformerv2

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@torch.no_grad()
def fusion_acc(scores, id2id, clip_num=1):
    assert scores.shape[0] == len(id2id.keys()) * clip_num
    ranks = np.zeros(scores.shape[0])
    for index, score in enumerate(scores):
        inds = np.argsort(score)[::-1]  # 排序后 得到排序后元素索引 最后一个即找到最大
        rank = 1e20  # init Score
        for i in id2id[index // clip_num]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    return r1, r5


# @torch.no_grad()
# def fusion_acc_backup(scores_r2t, scores_c2t, scores_mix, img2txt):
#     ranks = np.zeros(scores_r2t.shape[0])
#     for index, score in enumerate(scores_r2t):
#         inds = np.argsort(score)[::-1]  # 排序后 得到排序后元素索引 最后一个即找到最大
#         rank = 1e20  # init Score
#         for i in img2txt[index]:
#             tmp = np.where(inds == i)[0][0]
#             if tmp < rank:
#                 rank = tmp
#         ranks[index] = rank
#
#     r1_r2t = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#     r5_r2t = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#
#     return r1_r2t


def init_tokenizer(config):
    available_language_models = ["bert", "roberta"]
    assert config.FUSION.LANGUAGE_MODEL in available_language_models, f"Check language model name, should be one of the {available_language_models}"
    assert config.FUSION.LANGUAGE_TOKENIZER_DIR_PATH, f"Check language tokenizer path"

    if config.FUSION.LANGUAGE_MODEL == "bert":
        # tokenizer = BertTokenizer.from_pretrained('model/bert')
        tokenizer = BertTokenizer.from_pretrained(config.FUSION.LANGUAGE_TOKENIZER_DIR_PATH)

        # tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})  # 添加 vocab
        # tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  # 得到新添加的 vocab 的id

    else:
        raise NotImplementedError

    return tokenizer


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_input_tokens: int, output_dim: int = None):
        super().__init__()
        self.num_heads = num_heads

        self.positional_embedding = nn.Parameter(torch.randn(1, num_input_tokens + 1, embed_dim) * .02)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    #     self.initialize_parameters()
    #
    # def initialize_parameters(self):
    #     nn.init.normal_(self.positional_embedding, std=0.02)
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.trunc_normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0.)
    #             nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # N L+1 D
        x = x + self.positional_embedding.to(x.dtype)  #
        x = x.transpose(1, 0)  # [L, Batch_size, D]

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.transpose(1, 0)[:, 0, :]


@MODEL_REGISTRY.register()
class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.max_words = self.cfg.FUSION.MAX_WORDS
        self.tokenizer = init_tokenizer(cfg)

        # self.rgb_encoder = self.init_RGB_model().eval()
        # self.rgb_width = self.rgb_encoder.backbone.width
        self.rgb_width = cfg.FUSION.RGB_WIDTH
        # for param in self.rgb_encoder.parameters():
        #     param.requires_grad = False
        # logger.info("freeze vision encoder")

        # Coord Encoder
        self.coord_encoder = self.init_COORD_model()
        self.coord_width = self.coord_encoder.transformer_feature_dim
        self.load_pretrain(self.coord_encoder, checkpoint_model_key_name="model")  # 加载预训练权重

        for param in self.coord_encoder.parameters():  # Freeze coord encoder
            param.requires_grad = False
        self.coord_encoder = self.coord_encoder.eval()
        self.coord_encoder.train = disabled_train
        logger.info(f"freeze coord encoder")

        # 加载 语言模型配置
        assert os.path.isfile(self.cfg.FUSION.LANGUAGE_CONFIG_PATH), f"There is not config file at {self.cfg.FUSION.LANGUAGE_CONFIG_PATH}"
        if self.cfg.FUSION.LANGUAGE_MODEL == "bert":
            language_config = BertConfig.from_json_file(self.cfg.FUSION.LANGUAGE_CONFIG_PATH)
        elif self.cfg.FUSION.LANGUAGE_MODEL == "roberta":
            language_config = RobertaConfig.from_json_file(self.cfg.FUSION.LANGUAGE_CONFIG_PATH)
        else:
            raise NotImplementedError

        # custom config
        language_config.encoder_width = self.rgb_width  # cross attention 的 dim 采用rgb的dim
        language_config.add_cross_attention = True
        language_config.query_length = self.cfg.FUSION.NUM_QUERY_TOKEN

        # 初始化并加载语言模型权重
        assert os.path.isdir(self.cfg.FUSION.LANGUAGE_PRETRAIN_PATH) or os.path.isfile(self.cfg.FUSION.LANGUAGE_PRETRAIN_PATH) \
            , f"{self.cfg.FUSION.LANGUAGE_PRETRAIN_PATH} is wrong"
        if self.cfg.FUSION.LANGUAGE_MODEL == "bert":
            self.language_encoder = BertModel.from_pretrained(self.cfg.FUSION.LANGUAGE_PRETRAIN_PATH, config=language_config, add_pooling_layer=False)
        elif self.cfg.FUSION.LANGUAGE_MODEL == "roberta":
            self.language_encoder = RobertaModel.from_pretrained(self.cfg.FUSION.LANGUAGE_PRETRAIN_PATH, config=language_config, add_pooling_layer=False)
        else:
            raise f"Check LANGUAGE_MODEL"
        self.language_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_width = self.language_encoder.config.hidden_size

        # init Query
        self.query_tokens = nn.Parameter(torch.zeros(1, self.cfg.FUSION.NUM_QUERY_TOKEN, self.text_width))  # TODO 是否需要给 Query 加入位置编码
        self.query_tokens.data.normal_(mean=0.0, std=language_config.initializer_range)

        # 初始化 Query部分 额外的参数时，使用Bert模型的参数而不是随机初始化
        state_dict = self.language_encoder.state_dict()
        for name, param in self.language_encoder.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.rgb_ln = nn.LayerNorm(self.rgb_width)  # TODO 是否需要 LayerNorm 这个需要实验验证
        self.coord_ln = nn.LayerNorm(self.coord_width)

        # self.rgb_proj = nn.Linear(self.rgb_width, self.cfg.FUSION.EMBED_DIM)
        # self.coord_proj = nn.Linear(self.rgb_width, self.cfg.FUSION.EMBED_DIM)
        self.query_proj = nn.Linear(self.text_width, self.cfg.FUSION.EMBED_DIM)
        self.rgb_coord_uniform_proj = nn.Linear(self.coord_width, self.rgb_width)  # embeds 特征维度统一
        self.language_proj = nn.Linear(self.text_width, self.cfg.FUSION.EMBED_DIM)

        # For POS-RGB SA
        self.position_embedding = nn.Parameter(torch.randn(1, self.cfg.FUSION.NUM_COORD_RGB_TOKEN, self.rgb_width) * .02)
        self.type_embedding_layer = nn.Embedding(2, self.rgb_width)

        self.temp = nn.Parameter(0.07 * torch.ones([]))  # 温度系数
        self.vtm_head = nn.Linear(self.text_width, 2)

        # classification
        # self.sigmoid = nn.Sigmoid()
        # self.balance = nn.Parameter(torch.zeros(self.text_width))

        self.classifier = nn.Sequential(
            # TODO 线性分类器设计
            # nn.LayerNorm(self.text_width),
            # nn.Dropout(0.1),  # TODO 是否需要 Dropout？
            #
            # nn.Linear(self.text_width, self.text_width),
            # nn.GELU(),
            # nn.Linear(self.text_width, self.cfg.MODEL.NUM_CLASSES)

            # 2
            nn.Dropout(0.1),
            nn.Linear(self.text_width, self.cfg.MODEL.NUM_CLASSES)

            # 3
            # nn.Dropout(0.1),
            # nn.Linear(self.text_width * self.cfg.FUSION.NUM_QUERY_TOKEN, self.cfg.MODEL.NUM_CLASSES)
        )
        self.classifier[1].weight.data.normal_(mean=0.0, std=0.01)
        self.classifier[1].bias.data.zero_()

        # 4
        # self.classifier = AttentionPool(embed_dim=self.text_width, num_heads=12, num_input_tokens=self.cfg.FUSION.NUM_QUERY_TOKEN, output_dim=self.cfg.MODEL.NUM_CLASSES)

    #     self.initialize_parameters()
    #
    # def initialize_parameters(self):
    #     nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
    #     nn.init.normal_(self.type_embedding_layer.weight, std=0.02)
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.trunc_normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0.)
    #             nn.init.constant_(m.weight, 1.0)

    def one_stage(self, inputs, indexes, meta, alpha):
        # inputs: list [Tensor, ...] Tensor: [4, 3, 16, 224, 224]
        # indexes: Tensor [4]
        # meta: dict
        # -box_tensors: torch.Size([4, 32, 4, 4])
        # -box_categories: torch.Size([4, 32, 4])
        # -label: torch.Size([4])
        # -caption: (<class 'list'>, 4)
        # -text_id: torch.Size([4])

        device = torch.device("cuda")
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)  # 数值范围截断控制

        # rgb_ori = self.rgb_encoder(inputs)
        rgb_ori = inputs

        rgb_embeds = self.rgb_ln(rgb_ori)  # [B*NUM_SAMPLE 1+T 768]
        # rgb_embeds = rgb_embeds.unsqueeze(1)  # [B*NUM_SAMPLE 1 D] 维度增加 TODO：采用cls token的全局特征信息
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long, device=device)  # [B*NUM_SAMPLE 1+T]
        # rgb_feat = F.normalize(self.rgb_proj(rgb_embeds[:, 0, :]), dim=-1)  # [B*NUM_SAMPLE, 512]

        coord_embeds = self.coord_ln(self.coord_encoder(meta["box_categories"], meta["box_tensors"]))  # [8, 33, 512]
        coord_embeds = self.rgb_coord_uniform_proj(coord_embeds)
        coord_atts = torch.ones(coord_embeds.size()[:-1], dtype=torch.long, device=device)  # [8, 33]
        # coord_feat = F.normalize(self.coord_proj(coord_embeds[:, 0, :]), dim=-1)  # [8, 512]

        assert rgb_embeds.shape[0] == coord_embeds.shape[0], "The number of samples for RGB and COORD is not equal"

        # ======================== VTC ========================#
        # print(f"VTC")
        # Query-Vision
        query_tokens = self.query_tokens.expand(rgb_embeds.shape[0], -1, -1)

        # encoder_hidden_states = [coord_embeds, rgb_embeds]
        # encoder_attention_mask = [coord_atts, rgb_atts]
        encoder_hidden_states = torch.cat([coord_embeds, rgb_embeds], dim=1)
        encoder_attention_mask = torch.cat([coord_atts, rgb_atts], dim=1)
        encoder_hidden_states = self.coord_rgb_add_embedding(encoder_hidden_states, coord_embeds.shape[1], rgb_embeds.shape[1])
        # print(f"VTC: encoder_hidden_states: {encoder_hidden_states.shape}    encoder_attention_mask: {encoder_attention_mask.shape}")

        query_output = self.language_encoder(
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            # use_cache=True, # TODO 不使用cache
            return_dict=True,
        )
        query_feats = F.normalize(self.query_proj(query_output.last_hidden_state), dim=-1)  # [64, 32, 512]

        text = self.tokenizer(meta["caption"], padding='max_length', truncation=True, max_length=self.max_words, return_tensors="pt").to(device)
        text_output = self.language_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = F.normalize(self.language_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)  # [64, 512]

        idx = meta["text_id"].view(-1, 1)  # [32, 1]
        # idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, 32+queue_size]
        idx_all = concat_all_gather(idx) if self.cfg.NUM_GPUS > 1 else idx
        pos_idx = torch.eq(idx, idx_all.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

        if self.cfg.NUM_GPUS > 1:
            query_feats_all = concat_all_gather(query_feats)
            text_feat_all = concat_all_gather(text_feat)
        else:
            query_feats_all = query_feats
            text_feat_all = text_feat

        if self.cfg.FUSION.NUM_QUERY_TOKEN > 1:
            # [64, 1, 32, 512] @ [64, 512, 1] => [64, 64, 32, 1] => [64, 64, 32]
            sim_q2t = torch.matmul(query_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()  # [64, 64, 32]
            sim_q2t_max = sim_q2t.mean(dim=-1)
        else:
            sim_q2t = query_feats.squeeze() @ text_feat_all.t()
            sim_q2t_max = sim_q2t
        sim_q2t_max = sim_q2t_max / self.temp  # [64, 64]

        if self.cfg.FUSION.NUM_QUERY_TOKEN > 1:
            # [64, 1, 1, 512] @ [64, 512, 32] => [64, 64, 1, 32] => [64, 64, 32]
            sim_t2q = torch.matmul(text_feat.unsqueeze(1).unsqueeze(1), query_feats_all.permute(0, 2, 1)).squeeze()
            sim_t2q_max = sim_t2q.mean(dim=-1)
        else:
            sim_t2q = text_feat @ query_feats_all.squeeze().t()
            sim_t2q_max = sim_t2q
        sim_t2q_max = sim_t2q_max / self.temp  # [64, 64]

        # TODO: 验证 T2V 阶段是否有意义？
        # TODO: 这loss权重的分配暂时为均分，可能出现loss大的dominate其他loss。。。
        loss_q2t = -torch.sum(F.log_softmax(sim_q2t_max, dim=1) * sim_targets, dim=1).mean()
        loss_t2q = -torch.sum(F.log_softmax(sim_t2q_max, dim=1) * sim_targets, dim=1).mean()
        loss_vtc = (loss_q2t + loss_t2q) / 2

        # ======================== VTM ========================#
        # print(f"VTM")
        bs = rgb_embeds.shape[0]
        text_input_ids_world = concat_all_gather(text.input_ids) if self.cfg.NUM_GPUS > 1 else text.input_ids
        text_attention_mask_world = concat_all_gather(text.attention_mask) if self.cfg.NUM_GPUS > 1 else text.attention_mask
        rgb_embeds_world = concat_all_gather(rgb_embeds) if self.cfg.NUM_GPUS > 1 else rgb_embeds
        coord_embeds_world = concat_all_gather(coord_embeds) if self.cfg.NUM_GPUS > 1 else coord_embeds

        with torch.no_grad():
            mask = torch.eq(idx, idx_all.t())
            weights_q2t = F.softmax(sim_q2t_max, dim=1)
            weights_q2t.masked_fill_(mask, 0)

            weights_t2q = F.softmax(sim_t2q_max, dim=1)
            weights_t2q.masked_fill_(mask, 0)

        # Select negative samples
        rgb_embeds_neg = []
        coord_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2q[b], 1).item()
            rgb_embeds_neg.append(rgb_embeds_world[neg_idx])
            coord_embeds_neg.append(coord_embeds_world[neg_idx])

        rgb_embeds_neg = torch.stack(rgb_embeds_neg, dim=0)  # [64, 9, 768]
        coord_embeds_neg = torch.stack(coord_embeds_neg, dim=0)  # [64, 33, 768]

        # Select negative samples
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_q2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)  # [64, 30]
        text_atts_neg = torch.stack(text_atts_neg, dim=0)  # [64, 30]

        text_ids_all = torch.cat([text.input_ids, text.input_ids, text_ids_neg], dim=0)  # [192, 30]
        text_atts_all = torch.cat([text.attention_mask, text.attention_mask, text_atts_neg], dim=0)  # [192, 30]

        coord_embeds_all = torch.cat([coord_embeds, coord_embeds_neg, coord_embeds], dim=0)  # [192, 33, 768]
        coord_atts_all = torch.cat([coord_atts, coord_atts, coord_atts], dim=0)  # [192, 33]
        # print(f"is equal coord_atts_all: {torch.equal(coord_atts_all, torch.ones(coord_embeds_all.size()[:-1], dtype=torch.long).cuda())}")

        rgb_embeds_all = torch.cat([rgb_embeds, rgb_embeds_neg, rgb_embeds], dim=0)  # [192, 9, 768]
        rgb_atts_all = torch.cat([rgb_atts, rgb_atts, rgb_atts], dim=0)  # [192, 9]
        # print(f"is equal rgb_atts_all: {torch.equal(rgb_atts_all, torch.ones(rgb_embeds_all.size()[:-1], dtype=torch.long).cuda())}")

        query_tokens_vtm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)  # [192, 32, 768]
        query_atts_vtm = torch.ones(query_tokens_vtm.size()[:-1], dtype=torch.long, device=device)  # [192, 32]
        attention_mask_all = torch.cat([query_atts_vtm, text_atts_all], dim=1)

        # encoder_hidden_states = [coord_embeds_all, rgb_embeds_all]
        # encoder_attention_mask = [coord_atts_all, rgb_atts_all]
        encoder_hidden_states = torch.cat([coord_embeds_all, rgb_embeds_all], dim=1)
        encoder_attention_mask = torch.cat([coord_atts_all, rgb_atts_all], dim=1)
        encoder_hidden_states = self.coord_rgb_add_embedding(encoder_hidden_states, coord_embeds_all.shape[1], rgb_embeds_all.shape[1])
        # print(f"VTM: encoder_hidden_states: {encoder_hidden_states.shape}    encoder_attention_mask: {encoder_attention_mask.shape}")

        output_vtm = self.language_encoder(text_ids_all,
                                           query_embeds=query_tokens_vtm,
                                           attention_mask=attention_mask_all,
                                           encoder_hidden_states=encoder_hidden_states,
                                           encoder_attention_mask=encoder_attention_mask,
                                           return_dict=True,
                                           )

        vl_embeddings = output_vtm.last_hidden_state[:, :query_tokens_vtm.size(1), :]
        vl_output = self.vtm_head(vl_embeddings).mean(dim=1)  # TODO: 取平均的方式

        vtm_labels = torch.cat([torch.ones(bs, dtype=torch.long, device=device), torch.zeros(2 * bs, dtype=torch.long, device=device)], dim=0)
        loss_vtm = F.cross_entropy(vl_output, vtm_labels)

        # ======================== Classfication ======================== #
        # print(f"CLS")
        # Query-Vision
        query_tokens_cls = self.query_tokens.expand(rgb_embeds.shape[0], -1, -1)
        # encoder_hidden_states = [coord_embeds, rgb_embeds]
        # encoder_attention_mask = [coord_atts, rgb_atts]
        encoder_hidden_states = torch.cat([coord_embeds, rgb_embeds], dim=1)
        encoder_attention_mask = torch.cat([coord_atts, rgb_atts], dim=1)
        encoder_hidden_states = self.coord_rgb_add_embedding(encoder_hidden_states, coord_embeds.shape[1], rgb_embeds.shape[1])
        # print(f"CLS: encoder_hidden_states: {encoder_hidden_states.shape}    encoder_attention_mask: {encoder_attention_mask.shape}")
        cls_result = self.forward_classification(encoder_hidden_states, encoder_attention_mask, query_tokens_cls)

        return loss_vtc, loss_vtm, cls_result

    def coord_rgb_add_embedding(self, encoder_hidden_states, coord_token_len, rgb_token_len):
        assert self.cfg.FUSION.NUM_COORD_RGB_TOKEN == encoder_hidden_states.shape[1], f"self.cfg.FUSION.NUM_COORD_RGB_TOKEN={self.cfg.FUSION.NUM_COORD_RGB_TOKEN} != encoder_hidden_states.shape[1]={encoder_hidden_states.shape[1]}"

        encoder_hidden_states = encoder_hidden_states + self.position_embedding.to(encoder_hidden_states.dtype)  #
        type_ids = torch.cat([torch.zeros([1, coord_token_len], dtype=torch.long, device=encoder_hidden_states.device),
                              torch.ones([1, rgb_token_len], dtype=torch.long, device=encoder_hidden_states.device)], dim=1)
        type_embeddings = self.type_embedding_layer(type_ids)
        encoder_hidden_states = encoder_hidden_states + type_embeddings
        return encoder_hidden_states

    def forward_classification(self, encoder_hidden_states, encoder_attention_mask, query_tokens_cls):
        query_output_cls = self.language_encoder(
            query_embeds=query_tokens_cls,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            # use_cache=True, # TODO 不使用cache
            return_dict=True,
        )
        # query_feats = F.normalize(query_output.last_hidden_state, dim=-1)  # [32, 32, 768]
        query_feats_cls = query_output_cls.last_hidden_state

        # 1、特征信息取平均后 再根据平均后的特征信息 过分类层
        # cls_feat = query_feats_cls.mean(dim=1)  # [32, 768]  # TODO 取平均
        # # cls_feat, _ = query_feats_cls.max(dim=1)  # [32, 768]  # TODO 取max
        # cls_result = self.classifier(cls_feat)  # [32, 174]

        # 2、每个 query token 都经过分类层，然后将分类结果取平均
        cls_output = self.classifier(query_feats_cls)

        if self.cfg.FUSION.NUM_QUERY_TOKEN > 1:
            cls_result = cls_output.mean(dim=1)  # TODO 取平均/取max
        else:
            cls_result = cls_output.squeeze()

        # 3、query 表示每个bbox，全连接合并特征
        # query_feats_cls = rearrange(query_feats_cls, "B Q D -> B (Q D)")
        # cls_result = self.classifier(query_feats_cls)

        # 4、attention pooling
        # cls_result = self.classifier(query_feats_cls)

        return cls_result

    def forward(self, inputs, indexes, meta, alpha):
        # inputs: list [Tensor, ...] Tensor: [4, 3, 16, 224, 224]
        # indexes: Tensor [4]
        # meta: dict
        # -box_tensors: torch.Size([4, 32, 4, 4])
        # -box_categories: torch.Size([4, 32, 4])
        # -label: torch.Size([4])
        # -caption: (<class 'list'>, 4)
        # -text_id: torch.Size([4])

        return self.one_stage(inputs, indexes, meta, alpha)  # Use One stage method to train

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

        self.query_tokens_m = self.query_tokens_m * self.momentum + self.query_tokens * (1. - self.momentum)

    @torch.no_grad()
    # def _dequeue_and_enqueue(self, text_feat, idxs, fusion_feat=None):
    def _dequeue_and_enqueue(self, text_feat, idxs, rgb_feat=None, coord_feat=None):

        if self.cfg.NUM_GPUS > 1:
            text_feats = concat_all_gather(text_feat)
            rgb_feats = concat_all_gather(rgb_feat) if rgb_feat is not None else None
            coord_feats = concat_all_gather(coord_feat) if coord_feat is not None else None
        else:
            text_feats = text_feat
            rgb_feats = rgb_feat if rgb_feat is not None else None
            coord_feats = coord_feat if coord_feat is not None else None

        # assert rgb_feats.shape[0] == coord_feats.shape[0]
        assert rgb_feats is not None or coord_feats is not None, "rgb and coord features are both None"
        if rgb_feats is not None and coord_feats is not None:
            assert rgb_feats.shape[0] == coord_feats.shape[0], "the shapes of rgb and coord do not match"
        batch_size = coord_feats.shape[0] if coord_feats is not None else rgb_feats.shape[0]

        # batch_size = rgb_feats.shape[0]
        # batch_size = coord_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # rgb_queue coord_queue text_queue idx_queue ptr_queue
        if rgb_feats is not None:
            self.rgb_queue[:, ptr:ptr + batch_size] = rgb_feats.T
            # logger.info(f"update rgb feats queue")
        if coord_feats is not None:
            self.coord_queue[:, ptr:ptr + batch_size] = coord_feats.T
            # logger.info(f"update coord feats queue")
        self.language_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.ptr_queue[0] = ptr

    def init_COORD_model(self):
        model = VideoModelCoord(
            num_boxs=self.cfg.DATA.NUM_BBOXES,
            num_classes=self.cfg.MODEL.NUM_CLASSES,
            obj_class_num=self.cfg.FUSION.OBJ_CLASS_NUM,
            num_frames=self.cfg.FUSION.COORD_NUM_FRAME,
            coord_feature_dim=self.cfg.FUSION.COORD_FEATURE_DIM,

            spatial_transformer_depth=self.cfg.FUSION.COORD_NUM_SPATIAL_LAYER,
            temporal_transformer_depth=self.cfg.FUSION.COORD_NUM_TEMPORAL_LAYER
        )
        model.classifier = torch.nn.Identity()  # replace the classifier layer
        return model

    def init_RGB_model(self):
        model = Uniformerv2(self.cfg)
        model.backbone.transformer.proj = torch.nn.Identity()  # replace the classifier layer
        return model

    def load_pretrain(self, model: nn.Module, checkpoint_model_key_name='model'):
        """
        加载预训练模型
        """
        # assert model.__class__.__name__ in [self.rgb_encoder.__class__.__name__, self.coord_encoder.__class__.__name__]
        # if model.__class__.__name__ == self.rgb_encoder.__class__.__name__:
        #     model_path = self.cfg.FUSION.RGB_PRETRAIN_PATH
        # else:
        #     model_path = self.cfg.FUSION.COORD_PRETRAIN_PATH
        model_path = self.cfg.FUSION.COORD_PRETRAIN_PATH
        # model_path = self.cfg.FUSION.RGB_PRETRAIN_PATH

        assert os.path.isfile(model_path), f"No checkpoint is found in {model_path}"

        checkpoint = torch.load(model_path, map_location='cpu')

        if checkpoint_model_key_name in checkpoint.keys():
            pre_train_dict = checkpoint[checkpoint_model_key_name]
        else:
            pre_train_dict = checkpoint

        model_dict = model.state_dict()
        # Match pre-trained weights that have same shape as current model.
        pre_train_dict_match = OrderedDict()
        for k, v in pre_train_dict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                pre_train_dict_match[k] = v

        # Weights that do not have match from the pre-trained model.
        not_load_layers = [
            k
            for k in model_dict.keys()
            if k not in pre_train_dict_match.keys()
        ]
        if not_load_layers:
            for k in not_load_layers:
                logger.info("Network weights {} not loaded.".format(k))
        model.load_state_dict(pre_train_dict_match, strict=True)
        logger.info(f"load checkpoint from {model_path}")

    # def load_rgb_pretrain(self):
    #     """
    #     加载RGB预训练模型
    #     """
    #     assert self.cfg.FUSION.RGB_PRETRAIN_PATH, "rgb pretrain model path is wrong"
    #     assert os.path.isfile(self.cfg.FUSION.RGB_PRETRAIN_PATH), "No checkpoint found at '{}'".format(self.cfg.FUSION.RGB_PRETRAIN_PATH)
    #
    #     checkpoint = torch.load(self.cfg.FUSION.RGB_PRETRAIN_PATH, map_location='cpu')
    #     state_dict_3d = self.state_dict()
    #     for k in checkpoint.keys():
    #         if checkpoint[k].shape != state_dict_3d[k].shape:
    #             if len(state_dict_3d[k].shape) <= 2:
    #                 logger.info(f'Ignore: {k}')
    #                 continue
    #             logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
    #             time_dim = state_dict_3d[k].shape[2]
    #             checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)
    #     return checkpoint

    # def inflate_weight(self, weight_2d, time_dim, center=False):
    #     """
    #     将二维权重矩阵转换为三维权重矩阵
    #     """
    #     if center:
    #         weight_3d = torch.zeros(*weight_2d.shape)
    #         weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
    #         middle_idx = time_dim // 2
    #         weight_3d[:, :, middle_idx, :, :] = weight_2d
    #     else:
    #         weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
    #         weight_3d = weight_3d / time_dim
    #     return weight_3d
