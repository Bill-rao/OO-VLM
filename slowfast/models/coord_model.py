import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

# from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
#     resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked

# from model.r_transformer_3 import Transformer_Bolck

from .r_transformer import TransformerEncoder as Encoder
from .r_transformer import TransformerEncoderLayer as EncoderLayer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .build import MODEL_REGISTRY

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class RefNRIMLP(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
        super(RefNRIMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),  # 激活函数
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True)
        )
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x


class AMLP(nn.Module):
    def __init__(self, D_in, D_hid, D_out, act_layer=nn.ReLU, skip_connect=False, no_ln=False):
        super(AMLP, self).__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.fc_in = nn.Linear(D_in, D_hid)
        self.fc_out = nn.Linear(D_hid, D_out)

        self.fc_hid = nn.Sequential(
            nn.Linear(D_hid, D_hid),
            nn.ReLU()
        )
        self.ln = nn.LayerNorm(D_in) if not no_ln else nn.Identity()
        #     self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.ln(x)
        xs = self.fc_in(x)
        xs = self.act(xs)
        xs = self.fc_hid(xs)
        xs = self.fc_out(xs)
        # xs = self.fc1(x)
        # xs = self.act(xs)
        #
        # xs = self.fc2(xs)
        return xs


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}  # {'11': array([1., 0., 0.])}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # 将所有array([1., 0., 0.])，放到一个列表里面 [[1 0 0], [0 1 0], [0 0 1]]
    return labels_onehot


class VideoModelCoord(nn.Module):
    def __init__(self,
                 num_boxs: int,  # 锚框数量
                 num_classes: int,  # cls 分类结果数
                 obj_class_num: int,  # 锚框对应物体的类别总数
                 num_frames: int = 16,  # 帧数(max)
                 coord_feature_dim: int = 256,  # bbox坐标信息的特征维度
                 gnn_dropout=0.,
                 # Transformer
                 num_heads: int = 4,
                 spatial_transformer_depth: int = 1,
                 temporal_transformer_depth: int = 1,
                 tranformer_dropout=0.15,
                 mlp_ratio: float = 4.,
                 head_drop: float = 0.,
                 ):
        super(VideoModelCoord, self).__init__()
        self.nr_boxes = num_boxs  # 锚框数量
        self.num_classes = num_classes
        self.nr_frames = num_frames
        self.coord_feature_dim = coord_feature_dim
        self.obj_class_num = obj_class_num  # bbox obj 类别总数-noun类别总数   H2O:11 (0-8, 11),  Kitchen:300
        no_ln = False
        self.gnn_dropout = gnn_dropout

        # 根据 bbox 的数量决定GNN网络的边的数量
        if self.nr_boxes == 3:
            self.nr_edges = 6
        elif self.nr_boxes == 4:
            self.nr_edges = 12
        else:
            raise RuntimeError('num of nr_boxes error')

        # 用来给特征信息进行编码
        self.category_embed_layer = nn.Embedding(self.obj_class_num, self.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)  # kitchen_obj class_num + 1 = 301

        # 特征融合
        self.coord_category_fusion = nn.Sequential(
            nn.LayerNorm(self.coord_feature_dim // 2 * 3),
            nn.Linear(self.coord_feature_dim // 2 * 3, self.coord_feature_dim, bias=False),
            nn.ReLU(),
            # nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            # nn.ReLU()
        )

        # 将坐标信息转变为特征向量（coord_feature）的形式表示
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim // 2, bias=False),
            nn.ReLU()
        )

        # 将变化信息转换为特征向量
        self.flow_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim // 2, bias=False),
            nn.ReLU()
        )

        # 构造邻接矩阵 4*4 用来保存GNN结构信息
        # 4*4 意味着一个输入上最多只有4个node hand+3*objects
        # 邻接矩阵4个节点互相连接 无自连 双向连接
        edges = np.ones(self.nr_boxes) - np.eye(self.nr_boxes)
        self.send_edges = np.where(edges)[0]  # 邻接矩阵 每个元素行坐标
        self.recv_edges = np.where(edges)[1]  # 邻接矩阵 每个元素列坐标 邻接矩阵第一个元素 edges[send_edges[0], recv_edges[0]]
        # self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=True)
        self.edge2node_mat = nn.Parameter(torch.tensor(encode_onehot(self.recv_edges).transpose(), dtype=torch.float32), requires_grad=True)

        # RefNRIMLP 两层 Linear(in, hid) Linear(hid, out)
        # For GNN
        self.gnn_mlp1 = AMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, no_ln=no_ln)
        self.gnn_mlp2 = AMLP(self.coord_feature_dim, self.coord_feature_dim, self.coord_feature_dim, no_ln=no_ln)
        self.gnn_mlp3 = AMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, no_ln=no_ln)

        # # TODO：此处大约是需要重写的
        # if opt.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        '''
        bbox Transformer
        '''
        self.head_drop = nn.Dropout(head_drop)

        # Transformer Block init
        self.spatial_transformer_depth = spatial_transformer_depth  # 层数
        self.temporal_transformer_depth = temporal_transformer_depth  # 层数
        self.transformer_feature_dim = self.coord_feature_dim * 2  # transformer feature dim
        assert self.transformer_feature_dim % 64 == 0, "transformer feature dim error"

        # self.spatial_transformer_blocks = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=self.transformer_feature_dim,
        #         nhead=self.transformer_feature_dim // 64,
        #         dim_feedforward=int(self.transformer_feature_dim * mlp_ratio),
        #         dropout=tranformer_dropout,
        #         activation='gelu',
        #         batch_first=True
        #     ),
        #     num_layers=self.spatial_transformer_depth,
        #     norm=nn.LayerNorm(self.transformer_feature_dim)
        # )

        self.spatial_transformer_blocks = Encoder(
            encoder_layer=EncoderLayer(
                d_model=self.transformer_feature_dim,
                nhead=self.transformer_feature_dim // 64,
                dim_feedforward=int(self.transformer_feature_dim * mlp_ratio),
                dropout=tranformer_dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=self.spatial_transformer_depth,
            norm=nn.LayerNorm(self.transformer_feature_dim)
        )

        # self.spatial_transformer_blocks = nn.Sequential(*[Transformer_Bolck(
        #     dim=self.transformer_feature_dim,
        #     num_heads=self.transformer_feature_dim // 64,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=True,
        #     qk_norm=False,
        #     proj_drop=proj_drop,
        #     attn_drop=attn_drop,
        #     drop_path=-1, )
        #     for _ in range(self.spatial_transformer_depth)])

        # self.temporal_transformer_blocks = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=self.transformer_feature_dim,
        #         nhead=self.transformer_feature_dim // 64,
        #         dim_feedforward=int(self.transformer_feature_dim * mlp_ratio),
        #         dropout=tranformer_dropout,
        #         activation='gelu',
        #         batch_first=True
        #     ),
        #     num_layers=self.spatial_transformer_depth,
        #     norm=nn.LayerNorm(self.transformer_feature_dim)
        # )

        self.temporal_transformer_blocks = Encoder(
            encoder_layer=EncoderLayer(
                d_model=self.transformer_feature_dim,
                nhead=self.transformer_feature_dim // 64,
                dim_feedforward=int(self.transformer_feature_dim * mlp_ratio),
                dropout=tranformer_dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=self.spatial_transformer_depth,
            norm=nn.LayerNorm(self.transformer_feature_dim)
        )

        # self.temporal_transformer_blocks = nn.Sequential(*[Transformer_Bolck(
        #     dim=self.transformer_feature_dim,
        #     num_heads=self.transformer_feature_dim // 64,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=True,
        #     qk_norm=False,
        #     proj_drop=proj_drop,
        #     attn_drop=attn_drop,
        #     drop_path=-1, )
        #     for _ in range(self.temporal_transformer_depth)])

        scale = self.coord_feature_dim ** -0.5  # 系数 防止梯度爆炸（消失）问题
        self.spatial_feature_token = nn.Parameter(scale * torch.randn(1, 1, self.transformer_feature_dim))

        # self.v1_sptial_feature_token = nn.Parameter(torch.randn(1, 1, self.coord_feature_dim))
        # self.v2_sptial_feature_token = nn.Parameter(torch.randn(1, 1, self.coord_feature_dim))

        self.v1_temporal_feature_token = nn.Parameter(scale * torch.randn(1, 1, self.transformer_feature_dim))
        # self.p1_temporal_feature_token = nn.Parameter(scale * torch.randn(1, 1, self.coord_feature_dim))

        # TODO:可以试试用 nn.Embedding 来实现位置嵌入-类似bert
        self.spatial_embedding = nn.Parameter(torch.randn(1, self.nr_edges + self.nr_boxes, self.transformer_feature_dim) * .02)
        self.temporal_embedding = nn.Parameter(torch.randn(1, self.nr_frames, self.transformer_feature_dim) * .02)

        # self.cls_pos_embedding = nn.Parameter(torch.randn(1, 4, self.coord_feature_dim))

        # self.dropout = nn.Dropout(dropout)

        # self.spatial_transformer_blocks = nn.Sequential(*[Transformer_Bolck(
        #     dim=self.coord_feature_dim,
        #     num_heads=num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=True,
        #     qk_norm=False,
        #     proj_drop=proj_drop,
        #     attn_drop=attn_drop,
        #     drop_path=-1, )
        #     for i in range(self.spatial_transformer_depth)])
        # self.temporal_transformer_blocks = nn.Sequential(*[Transformer_Bolck(
        #     dim=self.coord_feature_dim,
        #     num_heads=num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=True,
        #     qk_norm=False,
        #     proj_drop=proj_drop,
        #     attn_drop=attn_drop,
        #     drop_path=-1, )
        #     for i in range(self.temporal_transformer_depth)])

        # 分类层
        self.classifier = nn.Sequential(
            # nn.Linear(self.coord_feature_dim * 2, 4 * self.coord_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(4 * self.coord_feature_dim, 2 * self.coord_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(2 * self.coord_feature_dim, self.num_classes)

            # nn.Linear(self.coord_feature_dim * 2, self.coord_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.coord_feature_dim, self.num_classes)

            nn.LayerNorm(self.transformer_feature_dim),
            # nn.Dropout(head_drop),
            nn.Linear(self.transformer_feature_dim, self.transformer_feature_dim),
            nn.ReLU(),
            nn.Linear(self.transformer_feature_dim, self.num_classes)
        )

        self.initialize_parameters()  # 初始化网络参数

        # NOTE
        # self.balance = nn.Parameter(torch.zeros((self.transformer_feature_dim)))
        # self.sigmoid = nn.Sigmoid()

        # if opt.fine_tune:
        #     self.fine_tune()

    def initialize_parameters(self):
        # nn.init.normal_(self.spatial_embedding, std=0.02)  # 从标注正态分布中提取值填入
        # nn.init.normal_(self.temporal_embedding, std=0.02)  # 从标注正态分布中提取值填入

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)
            else:
                for param in m.parameters():
                    nn.init.normal_(param, std=0.02)

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    def node2edge(self, node_embeddings):
        recv_embed = node_embeddings[:, self.recv_edges, :]
        send_embed = node_embeddings[:, self.send_edges, :]
        return torch.cat([send_embed, recv_embed], dim=3)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
            # print('incoming ', incoming.shape)
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming / (self.nr_boxes - 1)

    # def cls_transformer_forward(self, x_cls):
    #     x_cls = self.fc_norm(x_cls)
    #     x_cls = self.head_drop(x_cls)
    #     x_cls = self.classifier(x_cls)
    #     return x_cls  # [32, 174]  [32, 97]

    def get_flow(self, box_tensor: torch.Tensor):
        assert len(box_tensor.shape) == 4 and box_tensor.shape[1] == self.nr_frames and box_tensor.shape[2] == self.nr_boxes, "shape error, it should be (BS, T, N, C) tensor"
        BS, T, N, C = box_tensor.shape
        bbox_flow = torch.diff(box_tensor, dim=1)
        zeros = torch.zeros((BS, 1, N, C), device=bbox_flow.device, dtype=bbox_flow.dtype)
        bbox_flow = torch.cat([zeros, bbox_flow], dim=1)
        return bbox_flow

    def forward(self, box_categories, box_input, frame_mask=None):
        # box_input           torch.Size([64, 16, 4, 4])            [Batch, nr_frames, nr_boxes, 4(x1,x2,y1,y2)]
        # video_label         torch.Size([64])                      [Batch]
        # box_categories      torch.Size([64, 16, 4])               [Batch, nr_frames, box]
        # print('box_input 1', box_input.shape)
        # print('video_label', video_label.shape)
        # print('box_categories', box_categories.shape)

        b = box_input.shape[0]  # batch size
        # N：batchsize
        # E：edge数量
        # T：帧数frame
        # V：box数量=node
        N, E, T, V = b, self.nr_edges, self.nr_frames, self.nr_boxes  # TODO:注意这里边的数量要与node数量吻合

        # 计算得到flow信息
        box_flow = self.get_flow(box_input)
        # 重新调整tensor形状 b*物体框的数量*帧数 4坐标
        box_input = rearrange(box_input, 'b f n d -> (b f n) d')  # [3072, 4]
        box_flow = rearrange(box_flow, 'b f n d -> (b f n) d')

        box_categories = rearrange(box_categories, 'b f n -> (b f n)')
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*T*nr_b, coord_feature_dim//2)

        bbox_flow = self.flow_to_feature(box_flow)  # [bs*T*n, coord_feature_dim//2]
        bbox_coord = self.coord_to_feature(box_input)  # [bs*T*n, coord_feature_dim//2]

        bf = torch.cat([bbox_coord, bbox_flow, box_category_embeddings], dim=-1)  # (b*T*nr_b, D)
        bf = self.coord_category_fusion(bf)  # (b*T*nr_b, D)

        # 维度变化 [batch, frame, feature(position)] [64, 4, 16, 256]
        bf = rearrange(bf, '(b f n) d -> b n f d', b=N, f=T, n=V)  # [64, 3, 16, 256]

        '''
        GNN
        '''
        x_node_ori = bf  # 初始node信息

        # 节点特征转换为边特征 [batch*edge_num*frame, feature_dim*2] 双向信息
        x = self.node2edge(bf)
        x = x.reshape(N * E * T, -1)
        # 全连接 学习边特征
        x = self.gnn_mlp1(x)
        x = x.reshape(N, E, T, -1)  # [N, E, T, 256]
        x_edge_ori = x  # 初始edge信息

        # 节点特征转换为边特征
        x = self.edge2node(x)  # [N, V, T, D]
        x = x.reshape(N * V * T, -1)  # [N*V*T, D]
        # 全连接层 学习节点特征
        x = self.gnn_mlp2(x)
        x = x.reshape(N, V, T, -1)  # [N, V, T, D]
        x_node = torch.cat((x, x_node_ori), dim=-1)  # [N, V, T, D*2]

        # 节点特征转换为边特征
        x = self.node2edge(x)  # [N, E, T, D]
        x = x.reshape(N * E * T, -1)  # [N*E*T, D]
        # 全连接 学习边特征
        x = self.gnn_mlp3(x)
        x = x.reshape(N, E, T, -1)  # [N, E, T, D]
        x_edge = torch.cat((x, x_edge_ori), dim=-1)  # [N, E, T, D*2]

        # node edge 合并
        x = torch.cat((x_node, x_edge), dim=1)  # [N, V+E, T, D*2]
        # x = x.reshape(N * (V + E) * T, -1)
        # x = self.gnn_mlp4(x)
        # x = x.reshape(N, V + E, T, -1)  # [32, 4+12, 16, 256]
        # print('x 8', x.shape)

        '''
        空间 transformer
        '''
        x = rearrange(x, 'b n f d -> (b f) n d')  # [32*16, 4+12, 256]

        x = x + self.spatial_embedding[:, :, :]  # 添加 空间 位置编码 [32*16, 4+12, 256]
        # 添加token
        spatial_feature_tokens = repeat(self.spatial_feature_token, '1 1 d -> (b f) 1 d', b=N, f=T)  # [32*16, 1, 256]
        x = torch.cat([spatial_feature_tokens, x], dim=1)  # [Batch*Frames, edge_num+1, Dim] [32*16, 13, 256]

        # x = self.norm_pre(x)
        x = self.spatial_transformer_blocks(x)  # [32, 13, 256]
        # x = self.norm(x)

        x = x[:, 0]  # 提取空间特征信息 [32*16, 256]

        '''
        时间 transformer
        '''
        x = rearrange(x, '(b f) d -> b f d', b=N, f=T)  # [32, 16, 256]

        x = x + self.temporal_embedding[:, :, :]  # 添加 时间 位置编码 [32, 16, 256]

        # 时间 添加 token
        temporal_feature_tokens = repeat(self.v1_temporal_feature_token, '1 1 d -> b 1 d', b=N)  # [32, 1, 256]
        x = torch.cat([temporal_feature_tokens, x], dim=1)  # [Batch, T+1, Dim] [32*12, 17, 256]

        # 处理mask
        # frame_mask = torch.cat((torch.zeros([frame_mask.shape[0], temporal_feature_tokens.shape[1]], device='cuda'), frame_mask), dim=1)  # 将 cls token 的 mask 加上

        # x = self.norm_pre(x)
        # x = self.temporal_transformer_blocks(x, src_key_padding_mask=frame_mask)  # [32, 17, 256]
        x = self.temporal_transformer_blocks(x)  # [32*12, 17, 256]
        # x = self.norm(x)

        # x = x[:, 0]  # [32, 256]

        # cls = x[:, 0]

        # weight = self.sigmoid(self.balance)
        # residual = x[:, 1:].mean(1)  # [32 512]
        # cls = (1 - weight) * x[:, 0] + weight * residual
        # return self.classifier(cls)

        # 复制 vp 特征副本 开辟新内存空间并且独立于计算图之外 [32, 4, 256]
        # vp_feature = x.detach()
        # vp_feature = x.clone()

        # # cls
        # cls_result = self.cls_transformer_forward(x)
        # if len(cls_result) == 1:
        #     cls_result = cls_result.unsqueeze(0)
        #
        # return cls_result

        # return self.classifier(x[:, 0])  # 输出特征向量
        return x


@MODEL_REGISTRY.register()
class Coord(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.coord_model = VideoModelCoord(
            # image_size=args.img_size,
            # patch_size=args.patch_size,
            num_boxs=self.cfg.DATA.NUM_BBOXES,
            num_classes=self.cfg.MODEL.NUM_CLASSES,
            obj_class_num=self.cfg.FUSION.OBJ_CLASS_NUM,
            num_frames=self.cfg.FUSION.COORD_NUM_FRAME,
            # img_feature_dim=args.img_feature_dim,
            coord_feature_dim=self.cfg.FUSION.COORD_FEATURE_DIM,
            # num_heads=args.coord_feature_dim // 64,  # coord_feature_dim//64=num_heads
            spatial_transformer_depth=self.cfg.FUSION.COORD_NUM_SPATIAL_LAYER,
            temporal_transformer_depth=self.cfg.FUSION.COORD_NUM_TEMPORAL_LAYER
        )

    def forward(self, box_categories, box_input, frame_mask=None):
        output = self.coord_model(box_categories.to(torch.int64), box_input, frame_mask)
        return output
