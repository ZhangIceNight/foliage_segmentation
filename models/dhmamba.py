from typing import Optional, Tuple
import math
import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
from cvxpy.error import SolverError
from functools import partial
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from torch_cluster import knn

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from .pointnet2_utils import PointNetFeaturePropagation

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None, ):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


def get_density(xyz, avg_density):
    """
    计算点云的密度
    :param xyz: 点云坐标，形状为 (B, N, 3)
    :param avg_density: 平均密度，形状为 (B, 1)
    :return: 点云密度，形状为 (B, N, 1)
    """
    # 计算每个点的体积
    volume = torch.ones_like(xyz[..., :1])  # (B, N, 1)
    density = avg_density / volume  # (B, N, 1)
    return density

class Group(nn.Module):
    def __init__(self, num_group, group_size, avg_density=None):
        super().__init__()
        self.num_group = num_group # G
        self.group_size = group_size # M
        self.avg_density = avg_density # D
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def _get_neighborhood(self, xyz, centers, k):
        """
        xyz: [B, N, 3]
        centers: [B, G, 3]
        k: 邻居数
        return: neigh [B, G, k, 3]
        """
        B, N, _ = xyz.shape
        neigh_list = []
        for b in range(B):
            # knn(query, support, k) -> (idx_query, idx_support)
            idx_query, idx_support = knn(x=xyz[b], y=centers[b], k=k)

            # 按 query 分组，把 support 邻居取出来
            neigh = xyz[b][idx_support].view(centers[b].size(0), k, 3)
            neigh_list.append(neigh - centers[b].unsqueeze(1))  # 相对坐标
        return torch.stack(neigh_list, dim=0)  # [B, G, k, 3]


    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
       
        # knn to get the neighborhood
        neighborhood = self._get_neighborhood(xyz, center, self.group_size)

        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        # first conv
        feature = self.first_conv(point_groups.transpose(2, 1))  # bs*g, 256, n
        
        # global feature
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # bs*g, 256, 1

        # feature expansion
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # bs*g, 512, n

        # second conv
        feature = self.second_conv(feature)  # bs*g, encoder_channel, n

        # global feature
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # bs*g, encoder_channel

        
        return feature_global.reshape(bs, g, self.encoder_channel)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            init.constant_(self.bias.data, 0.1)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN_layer(nn.Module):
    def __init__(self, img_len, in_c):
        super(HGCN_layer, self).__init__()
        self.gc1 = GraphConvolution(in_c, in_c)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)

        self.gc2 = GraphConvolution(in_c, in_c)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)

        self.gc3 = GraphConvolution(in_c, in_c)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(feature + gc1)

        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(feature + gc2)

        gc3 = self.gc3(gc2, H)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(feature + gc3)
        return gc3


class HGCNNet_deep(nn.Module):
    def __init__(self, img_len):
        super(HGCNNet, self).__init__()
        self.gc1 = GraphConvolution(384, 384)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer1 = HGCN_layer(img_len, 384)

        self.gc2 = GraphConvolution(384, 384)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer2 = HGCN_layer(img_len, 384)

        self.gc3 = GraphConvolution(384, 384)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer3 = HGCN_layer(img_len, 384)

        self.gc4 = GraphConvolution(384, 384)
        self.bn4 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer4 = HGCN_layer(img_len, 384)

        self.gc5 = GraphConvolution(384, 384)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc1 = self.HGCN_layer1(gc1, H)

        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc2 = self.HGCN_layer2(gc2, H)

        gc3 = self.gc3(gc2, H)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)
        gc3 = self.HGCN_layer3(gc3, H)

        gc4 = self.gc4(gc3, H)
        gc4 = self.bn4(gc4)
        gc4 = self.relu(gc4)
        gc4 = self.HGCN_layer4(gc4, H)

        gc5 = self.gc5(gc4, H)
        gc5 = self.relu(gc5)
        return gc5


class HGCNNet(nn.Module):
    def __init__(self, img_len):
        super(HGCNNet, self).__init__()
        # 第一层
        self.gc1 = GraphConvolution(384, 384)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer1 = HGCN_layer(img_len, 384)

        # 第二层
        self.gc2 = GraphConvolution(384, 384)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer2 = HGCN_layer(img_len, 384)

        # 输出层
        self.gc3 = GraphConvolution(384, 384)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        # 第一层
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc1 = self.HGCN_layer1(gc1, H)

        # 第二层
        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc2 = self.HGCN_layer2(gc2, H)

        # 输出层
        gc3 = self.gc3(gc2, H)
        gc3 = self.relu(gc3)
        
        return gc3

class MixerModelForSegmentation(MixerModel):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.fetch_idx = fetch_idx

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        feature_list = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if idx in self.fetch_idx:
                if not self.fused_add_norm:
                    residual_output = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states_output = self.norm_f(residual_output.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states_output = fused_add_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                feature_list.append(hidden_states_output)
        return feature_list


class DHMamba(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.cls_dim = num_classes
        self.group_size = 32
        self.num_group = 128
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # Weight for hypergraph merging
        self.W = Parameter(torch.ones(self.num_group * 3))
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.HGCN = HGCNNet(img_len=self.num_group)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=False,
                                                drop_path=0.2,
                                                fetch_idx=[3, 7, 11])

        self.drop_out = nn.Dropout(0)
        self.drop_path_rate = 0.1
        self.drop_path_block = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024])
        
        # 更新卷积层的输入维度
        self.convs1 = nn.Conv1d(3328, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print('missing_keys')
            if incompatible.unexpected_keys:
                print('unexpected_keys')
            print(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print(f'[Mamba] No ckpt is loaded, training from scratch!')

    def KNN(self, X, n_neighbors, is_prob=True, dist=None):
        """
        torch 实现的 KNN 图构建
        X: torch.Tensor, shape [N, D], 节点特征
        n_neighbors: int, 邻居数
        is_prob: bool, 是否用高斯权重，否则是0-1
        return:
            knn: torch.Tensor, shape [N, N]
        """
        device = X.device
        N = X.size(0)

        # pairwise 距离 (欧式)
        # dist = torch.cdist(X, X, p=2)  # [N, N]

        # 取每个节点 top-k 最近邻 (包含自己，因为 dist[i,i]=0 最小)
        knn_val, knn_idx = torch.topk(dist, k=n_neighbors+1, dim=1, largest=False)

        # 构造稠密邻接矩阵
        row_idx = torch.arange(N, device=device).unsqueeze(1).repeat(1, n_neighbors+1).reshape(-1)
        col_idx = knn_idx.reshape(-1)

        if not is_prob:
            values = torch.ones_like(row_idx, dtype=torch.float32, device=device)
        else:
            avg_dist = dist.mean()
            values = torch.exp(- (knn_val.reshape(-1) ** 2) / (avg_dist ** 2 + 1e-8))

        knn = torch.zeros((N, N), device=device)
        knn[row_idx, col_idx] = values

        # 保证每个节点至少和自己相连（对角线 = 1）
        knn.fill_diagonal_(1.0)

        return knn

    def similarity(self, X, n_neighbors):
        """
        X: torch.Tensor, shape [N, D]，N个节点，每个节点D维特征
        n_neighbors: int, top-k相似节点数量
        返回:
            sim: torch.Tensor, shape [N, N]，0-1矩阵表示超图连接
        """
        # L2归一化
        X_norm = X / X.norm(dim=1, keepdim=True)  # [N, D]
        
        # 相似度矩阵 (余弦相似度)
        sim_mat = X_norm @ X_norm.T  # [N, N]
        
        # 取每行 top-k 索引 (包含自己)
        topk_vals, topk_idx = torch.topk(sim_mat, k=n_neighbors+1, dim=1, largest=True)
        
        # 构建 sim 矩阵
        N = X.shape[0]
        sim = torch.zeros(N, N, device=X.device, dtype=torch.float32)
        row_idx = torch.arange(N, device=X.device).unsqueeze(1).expand(-1, n_neighbors+1)  # [N, k+1]
        sim[row_idx, topk_idx] = 1.0
        
        # 强制对角线为1，保证每个节点自己被选上
        sim.fill_diagonal_(1.0)
        
        return sim

    def hyperG(self, knn, l1, sim, W, device=None):
        """
        knn, l1, sim: torch.Tensor, shape (N, E_knn / E_l1 / E_sim)，0/1 或权重矩阵
        W: torch.Tensor, shape (E_total,) or (E_total, E_total)，超边权重
        return: G (torch.Tensor), shape (N, N)
        """

        if device is None:
            device = knn.device

        # 拼接超图关联矩阵 H: [N, E]
        H = torch.cat((knn, l1, sim), dim=1).to(device)   # [N, E]

        # 度矩阵（节点和超边）
        DV = torch.sum(H, dim=1)          # [N]
        DE = torch.sum(H, dim=0)          # [E]

        # 构造对角矩阵的逆/平方逆
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -0.5))

        # H^T
        HT = H.t()

        # 中间矩阵
        DV2_H = DV2 @ H                   # [N, E]
        invDE_HT_DV2 = invDE @ HT @ DV2   # [E, N]

        # 权重矩阵 w
        if W.dim() == 1:
            W = torch.diag(W)
        w = W.to(device)

        # 组合得到 G
        G = DV2_H @ (w @ invDE_HT_DV2)    # [N, N]
        return G
    
    def abhyperG(self, hyp, W):
        H = hyp
        # H = knn
        # the degree of the node
        DV = np.sum(H, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        HT = H.T
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        DV2_H = torch.as_tensor(DV2_H).cuda().float()
        invDE_HT_DV2 = torch.as_tensor(invDE_HT_DV2).cuda().float()

        w = torch.diag(W)
        G = torch.mm(w, invDE_HT_DV2)
        G = torch.mm(DV2_H, G)
        return G
    
    def l1_representation(self, X, n_neighbors, reg_eps=1e-3, device=None, dist=None):
        """
        Torch 版本的 L1-like (这里使用 L2/LLE 闭式解) 局部重建权重。
        Args:
            X: torch.Tensor, shape (n_nodes, feat_dim), dtype=float32/64, on some device
            n_neighbors: int, number of neighbors to use (exclude self)
            reg_eps: float, regularization coefficient multiplier for trace(C)
            device: optional device (if None, 使用 X.device)
        Returns:
            weights: torch.Tensor, shape (n_nodes, n_nodes), dtype same as X,
                    weights[j, i] = weight of neighbor j for reconstructing node i.
                    Columns sum to 1 (except columns that remain zero if something weird).
        Notes:
            - Complexity O(n_nodes * k^2 * d) per sample; 对 n_nodes ~ 128, k~8..32 很快。
            - 如果想要稀疏 COO 输出可以再把非零元素取出来。
        """
        if device is None:
            device = X.device
        X = X.to(device)

        n_nodes, feat_dim = X.shape
        k = int(n_neighbors)
        assert k >= 1 and k < n_nodes, "n_neighbors must be >=1 and < n_nodes"

        # 距离矩阵并取 topk（包含 self）
        # dist = torch.cdist(X, X)  # (n_nodes, n_nodes)
        # 取 k+1 个最近（包含自己），然后排除自己
        _, knn_idx_all = torch.topk(dist, k=k+1, largest=False, sorted=False)  # (n_nodes, k+1)
        knn_idx = knn_idx_all[:, 1:k+1]  # (n_nodes, k) 排除了自己

        # 预分配 weight 矩阵
        weights = torch.zeros((n_nodes, n_nodes), device=device, dtype=X.dtype)

        ones_k = torch.ones((k,), device=device, dtype=X.dtype)

        eye_k = torch.eye(k, device=device, dtype=X.dtype)

        for i in range(n_nodes):
            neighbors = knn_idx[i]            # shape (k,)
            P = X[neighbors]                  # (k, d)
            v = X[i].unsqueeze(0)             # (1, d)
            Z = P - v                         # (k, d)

            # C = Z Z^T  (k x k)
            C = Z @ Z.transpose(0, 1)         # (k, k)

            # regularize: add eps * trace(C) to diagonal (标准做法)
            traceC = torch.trace(C)
            # 若 traceC == 0（所有邻居与中心相同）也要加一个小值
            reg = reg_eps * (traceC if traceC > 0 else 1.0) + 1e-6
            C = C + eye_k * reg

            # solve C w = 1
            try:
                w = torch.linalg.solve(C, ones_k)    # (k,)
            except RuntimeError:
                # 万一奇异，用伪逆回退
                w = torch.matmul(torch.linalg.pinv(C), ones_k)

            s = w.sum()
            if s.abs() < 1e-12:
                # 如果求解出全零（极端情况），退化到均匀权重
                w = torch.ones_like(w) / k
            else:
                w = w / s

            # 填入 weights 的第 i 列（neighbors 行）
            weights[neighbors, i] = w

        return weights  # torch.Tensor, shape (n_nodes, n_nodes)


    def forward(self, pts):
        B, N, C = pts.shape
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)    # neighborhood: [B, G, M, 3] (G: num_group=128, M: group_size=32)
                                                        # center: [B, G, 3]
        group_input_tokens = self.encoder(neighborhood)  # [B, G, encoder_dims=384]

        pos = self.pos_embed(center) # [B, G, trans_dim=384]

        # hypergraph serailization
        X = group_input_tokens
        H = []
        n_neighbors = 4
        for j in range(B):
            Xj = X[j, :, :]
            dist = torch.cdist(Xj, Xj, p=2)  # [N, N]
            # 3种超图构建方式
            knn = self.KNN(Xj, n_neighbors, dist=dist)
            l1 = self.l1_representation(Xj, n_neighbors, dist=dist)
            sim = self.similarity(Xj, n_neighbors)

            G = self.hyperG(knn, l1, sim, self.W)

            H.append(G.unsqueeze(0))

        H = torch.cat(H, dim=0) # [B, 3G, 3G]

        group_input_tokens = self.HGCN(group_input_tokens, H) # [B, G, 384]


        # final input
        x = group_input_tokens # [B, G, 384]

        feature_list = self.blocks(x, pos) # List of 3 tensors, each [B, G, 384]

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list] # List of 3 tensors, each [B, 384, G]
        x = torch.cat((feature_list), dim=1)  # 1152 # [B, 1152, G]  (384*3 = 1152)
        x_max = torch.max(x, 2)[0] # [B, 1152]
        x_avg = torch.mean(x, 2) # [B, 1152]
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N) # [B, 1152, N]
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N) # [B, 1152, N]


        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x) # [B, 3328, N]

        x = torch.cat((f_level_0, x_global_feature), 1)  # [B, 3328, N]
        x = self.relu(self.bns1(self.convs1(x)))  # [B, 512, N]
        x = self.dp1(x)  # [B, 512, N]
        x = self.relu(self.bns2(self.convs2(x)))  # [B, 256, N]
        x = self.convs3(x)  # [B, cls_dim, N]

        return x

    
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
