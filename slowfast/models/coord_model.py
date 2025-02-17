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
            nn.ELU(inplace=True),
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
                 num_boxs: int,
                 num_classes: int,
                 obj_class_num: int,
                 num_frames: int = 16,
                 coord_feature_dim: int = 256,
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
        self.nr_boxes = num_boxs
        self.num_classes = num_classes
        self.nr_frames = num_frames
        self.coord_feature_dim = coord_feature_dim
        self.obj_class_num = obj_class_num
        no_ln = False
        self.gnn_dropout = gnn_dropout

        if self.nr_boxes == 3:
            self.nr_edges = 6
        elif self.nr_boxes == 4:
            self.nr_edges = 12
        else:
            raise RuntimeError('num of nr_boxes error')

        self.category_embed_layer = nn.Embedding(self.obj_class_num, self.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)  # kitchen_obj class_num + 1 = 301

        self.coord_category_fusion = nn.Sequential(
            nn.LayerNorm(self.coord_feature_dim // 2 * 3),
            nn.Linear(self.coord_feature_dim // 2 * 3, self.coord_feature_dim, bias=False),
            nn.ReLU(),
            # nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            # nn.ReLU()
        )

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim // 2, bias=False),
            nn.ReLU()
        )

        self.flow_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim // 2, bias=False),
            nn.ReLU()
        )

        edges = np.ones(self.nr_boxes) - np.eye(self.nr_boxes)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        # self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=True)
        self.edge2node_mat = nn.Parameter(torch.tensor(encode_onehot(self.recv_edges).transpose(), dtype=torch.float32), requires_grad=True)

        # For GNN
        self.gnn_mlp1 = AMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, no_ln=no_ln)
        self.gnn_mlp2 = AMLP(self.coord_feature_dim, self.coord_feature_dim, self.coord_feature_dim, no_ln=no_ln)
        self.gnn_mlp3 = AMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, no_ln=no_ln)

        '''
        bbox Transformer
        '''
        self.head_drop = nn.Dropout(head_drop)

        # Transformer Block init
        self.spatial_transformer_depth = spatial_transformer_depth
        self.temporal_transformer_depth = temporal_transformer_depth
        self.transformer_feature_dim = self.coord_feature_dim * 2  # transformer feature dim
        assert self.transformer_feature_dim % 64 == 0, "transformer feature dim error"

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

        scale = self.coord_feature_dim ** -0.5
        self.spatial_feature_token = nn.Parameter(scale * torch.randn(1, 1, self.transformer_feature_dim))

        # self.v1_sptial_feature_token = nn.Parameter(torch.randn(1, 1, self.coord_feature_dim))
        # self.v2_sptial_feature_token = nn.Parameter(torch.randn(1, 1, self.coord_feature_dim))

        self.v1_temporal_feature_token = nn.Parameter(scale * torch.randn(1, 1, self.transformer_feature_dim))
        # self.p1_temporal_feature_token = nn.Parameter(scale * torch.randn(1, 1, self.coord_feature_dim))

        # TODO:可以试试用 nn.Embedding 来实现位置嵌入-类似bert
        self.spatial_embedding = nn.Parameter(torch.randn(1, self.nr_edges + self.nr_boxes, self.transformer_feature_dim) * .02)
        self.temporal_embedding = nn.Parameter(torch.randn(1, self.nr_frames, self.transformer_feature_dim) * .02)

        # 分类层
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.transformer_feature_dim),
            # nn.Dropout(head_drop),
            nn.Linear(self.transformer_feature_dim, self.transformer_feature_dim),
            nn.ReLU(),
            nn.Linear(self.transformer_feature_dim, self.num_classes)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
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
        b = box_input.shape[0]  # batch size
        N, E, T, V = b, self.nr_edges, self.nr_frames, self.nr_boxes

        box_flow = self.get_flow(box_input)
        box_input = rearrange(box_input, 'b f n d -> (b f n) d')
        box_flow = rearrange(box_flow, 'b f n d -> (b f n) d')

        box_categories = rearrange(box_categories, 'b f n -> (b f n)')
        box_category_embeddings = self.category_embed_layer(box_categories)

        bbox_flow = self.flow_to_feature(box_flow)
        bbox_coord = self.coord_to_feature(box_input)

        bf = torch.cat([bbox_coord, bbox_flow, box_category_embeddings], dim=-1)
        bf = self.coord_category_fusion(bf)

        bf = rearrange(bf, '(b f n) d -> b n f d', b=N, f=T, n=V)
        '''
        GNN
        '''
        x_node_ori = bf

        x = self.node2edge(bf)
        x = x.reshape(N * E * T, -1)
        x = self.gnn_mlp1(x)
        x = x.reshape(N, E, T, -1)
        x_edge_ori = x

        x = self.edge2node(x)
        x = x.reshape(N * V * T, -1)
        x = self.gnn_mlp2(x)
        x = x.reshape(N, V, T, -1)
        x_node = torch.cat((x, x_node_ori), dim=-1)

        x = self.node2edge(x)
        x = x.reshape(N * E * T, -1)
        x = self.gnn_mlp3(x)
        x = x.reshape(N, E, T, -1)
        x_edge = torch.cat((x, x_edge_ori), dim=-1)

        x = torch.cat((x_node, x_edge), dim=1)  # [N, V+E, T, D*2]

        '''
        Spatial transformer
        '''
        x = rearrange(x, 'b n f d -> (b f) n d')

        x = x + self.spatial_embedding[:, :, :]
        spatial_feature_tokens = repeat(self.spatial_feature_token, '1 1 d -> (b f) 1 d', b=N, f=T)
        x = torch.cat([spatial_feature_tokens, x], dim=1)
        x = self.spatial_transformer_blocks(x)

        x = x[:, 0]

        '''
        Temporal transformer
        '''
        x = rearrange(x, '(b f) d -> b f d', b=N, f=T)

        x = x + self.temporal_embedding[:, :, :]

        temporal_feature_tokens = repeat(self.v1_temporal_feature_token, '1 1 d -> b 1 d', b=N)
        x = torch.cat([temporal_feature_tokens, x], dim=1)

        x = self.temporal_transformer_blocks(x)  # [32*12, 17, 256]

        # return self.classifier(x[:, 0])
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
