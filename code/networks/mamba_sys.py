import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

### ----新增代码：边界检测模块（方案V4）----
class EdgeDetector(nn.Module):
    """
    边界检测模块 - 用于提取特征图的边界信息
    使用 Sobel 算子进行边界检测
    """
    def __init__(self, in_channels):
        super().__init__()
        # Sobel 算子卷积核
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        
        # 初始化 Sobel 卷积核
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 为每个通道复制卷积核
        for i in range(in_channels):
            self.sobel_x.weight.data[i, 0] = sobel_kernel_x
            self.sobel_y.weight.data[i, 0] = sobel_kernel_y
        
        # 冻结参数（Sobel核不需要学习）
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
        # 边界特征增强网络
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - 输入特征图
        Returns:
            edge_feat: (B, C, H, W) - 边界特征图
        """
        # 计算梯度
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        
        # 梯度幅值 (边界强度)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # 增强边界特征
        edge_feat = self.edge_enhance(edge_magnitude)
        
        return edge_feat


### ----Cross-Attention 模块（方案A/V3）----
class CrossAttention(nn.Module):
    """
    Cross-Attention 模块用于方案A（双流架构）
    Q 来自 register tokens, K/V 来自 patch tokens
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, query, key_value):
        """
        Args:
            query: register tokens (B, num_cls, C)
            key_value: patch features (B, H, W, C) 或 (B, N, C)
        Returns:
            updated query (B, num_cls, C)
        """
        B, num_cls, C = query.shape
        
        # 如果 key_value 是 4D (B, H, W, C)，flatten 成 (B, H*W, C)
        if key_value.dim() == 4:
            B_kv, H, W, C_kv = key_value.shape
            key_value = key_value.view(B_kv, H * W, C_kv)  # (B, N, C)
        
        N = key_value.shape[1]
        
        # Norm
        query = self.norm1(query)
        key_value = self.norm2(key_value)
        
        # Q from register, K/V from patches
        q = self.q(query).reshape(B, num_cls, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, N, head_dim)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, num_cls, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Aggregate
        x = (attn @ v).transpose(1, 2).reshape(B, num_cls, C)  # (B, num_cls, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

### ----Register Token 索引生成----
def get_cls_idx(H, W, n_cls):
    """
    生成 register token 插入位置的索引
    
    Args:
        H, W: grid 高度和宽度
        n_cls: register token 数量
    
    Returns:
        token_idx: 重排后的 token 索引
        cls_pos: register token 在重排后序列中的位置
    """
    n_tokens = H * W
    L = n_tokens // (n_cls + 1)  # 每段 patch 数量
    
    # 构造 [L个patch][REG1][L个patch][REG2]... 的索引
    token_idx = torch.cat([
        torch.arange(L * n_cls).view(n_cls, -1),  # patch tokens
        torch.arange(n_tokens, n_tokens + n_cls).unsqueeze(-1)  # register tokens
    ], dim=1).contiguous().view(-1)
    
    # 添加剩余未分配的 patch tokens
    token_idx = torch.cat([token_idx, torch.arange(L * n_cls, n_tokens)])
    
    # 计算 register token 在重排后序列中的位置
    cls_pos = torch.arange(L, L * (n_cls + 1) + n_cls, L + 1)
    
    return token_idx, cls_pos
### ----新增代码----

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    # def forward(self, x):
    #     x = self.proj(x).permute(0, 2, 3, 1)
    #     if self.norm is not None:
    #         x = self.norm(x)
    #     return x

    def forward(self, x):
        # 新增：记录输入图像尺寸
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W')
        
        # 新增：计算 grid_size
        self.grid_size = (x.size(2), x.size(3))  # (H', W')
        
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)

        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x

class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        Upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.upsample is not None:
            x = self.upsample(x)

        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, num_classes=4, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", num_cls_tokens=12, use_register=True, 
                 register_version='cross_attn', **kwargs):  # 新增 register_version 参数
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.num_features_up = int(dims[0] * 2)
        self.dims = dims
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        # ---新增：Register token 支持---------
        self.num_cls_tokens = num_cls_tokens
        self.use_register = use_register
        self.register_version = register_version  # 'cross_attn' 或 'pooled'
        
        if self.use_register and self.num_cls_tokens > 0:
            # === 共用模块：Register token 参数 ===
            self.cls_token = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
            
            # Register 特征的线性投影层（用于融合到解码器）
            self.register_proj = nn.Linear(self.embed_dim * num_cls_tokens, self.embed_dim)
            
            # 门控融合层（解码器融合时使用）
            self.register_gate = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Sigmoid()
            )
            
            # 改进的初始化
            trunc_normal_(self.cls_token, std=0.02)
            trunc_normal_(self.pos_embed_cls, std=0.02)
            
            # === 方案A：Cross-Attention 逐层更新（推荐）===
            if self.register_version == 'cross_attn':
                print(f">>> Using Register Version: Cross-Attention (方案A)")
                # 为每个编码器层创建一个 CrossAttention 模块
                self.cross_attn_layers = nn.ModuleList([
                    CrossAttention(
                        dim=self.embed_dim,  # 所有 cross-attn 都在 embed_dim 空间
                        num_heads=8,
                        qkv_bias=True,
                        attn_drop=attn_drop_rate,
                        proj_drop=drop_rate
                    )
                    for _ in range(len(depths))
                ])
                # 通道投影层：将不同层的输出投影到 embed_dim
                self.register_projectors = nn.ModuleList([
                    nn.Linear(self.dims[min(i+1, len(self.dims)-1)], self.embed_dim) 
                    for i in range(len(self.dims))
                ])
            
            # === 方案D：Pooled Aggregation + Gated Update（备用）===
            elif self.register_version == 'pooled':
                print(f">>> Using Register Version: Pooled Aggregation (方案D)")
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.register_updater = nn.Sequential(
                    nn.Linear(self.embed_dim * (1 + num_cls_tokens), self.embed_dim * num_cls_tokens),
                    nn.LayerNorm(self.embed_dim * num_cls_tokens),
                    nn.GELU(),
                    nn.Linear(self.embed_dim * num_cls_tokens, self.embed_dim * num_cls_tokens),
                    nn.Sigmoid()
                )
                self.layer_scales = nn.Parameter(torch.ones(len(depths)))
                self.global_projectors = nn.ModuleList([
                    nn.Linear(self.dims[min(i+1, len(self.dims)-1)], self.embed_dim) 
                    for i in range(len(self.dims))
                ])
            
            # === 方案V4：Cross-Attention + 边界增强门控（最优融合）===
            elif self.register_version == 'enhanced_gate':
                print(f">>> Using Register Version: Enhanced Gated Fusion (方案V4)")
                print(f"    Features: Cross-Attention + Dual-Channel Gating + Boundary Enhancement")
                
                # Cross-Attention 层（继承方案A）
                self.cross_attn_layers = nn.ModuleList([
                    CrossAttention(
                        dim=self.embed_dim,
                        num_heads=8,
                        qkv_bias=True,
                        attn_drop=attn_drop_rate,
                        proj_drop=drop_rate
                    )
                    for _ in range(len(depths))
                ])
                
                # 通道投影层
                self.register_projectors = nn.ModuleList([
                    nn.Linear(self.dims[min(i+1, len(self.dims)-1)], self.embed_dim) 
                    for i in range(len(self.dims))
                ])
                
                # 边界检测器
                self.edge_detector = EdgeDetector(in_channels=self.embed_dim)
                
                # 双通道门控网络
                # 全局信息门控（控制 Register 全局特征的权重）
                self.global_gate = nn.Sequential(
                    nn.Linear(self.embed_dim * 2, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Sigmoid()
                )
                
                # 边界信息门控（控制边界特征的权重）
                self.boundary_gate = nn.Sequential(
                    nn.Linear(self.embed_dim * 2, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Sigmoid()
                )
                
                # 融合权重（可学习的全局和边界平衡系数）
                self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # [global, boundary]
                
            else:
                raise ValueError(f"Unknown register_version: {register_version}, must be 'cross_attn', 'pooled', or 'enhanced_gate'")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        ### -------------------------------------
        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                # dim=dims[i_layer], #int(embed_dim * 2 ** i_layer)
                dim = int(dims[0] * 2 ** i_layer),
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(dims[0]*2**(self.num_layers-1-i_layer)),
            int(dims[0]*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = VSSLayer_up(
                    dim= int(dims[0] * 2 ** (self.num_layers-1-i_layer)),
                    depth=depths[(self.num_layers-1-i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=4,dim=self.embed_dim)
            self.output = nn.Conv2d(in_channels=self.embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)
        
        # ===== Deep Supervision: 辅助分割头 =====
        # 为decoder每一层添加辅助分割头
        self.use_deep_supervision = True  # 是否启用Deep Supervision
        if self.use_deep_supervision:
            self.aux_heads = nn.ModuleList()
            # Decoder各层维度（layer_up之后的维度）: [384, 192, 96, 96]
            # Layer 0: 768 -> (PatchExpand) -> 384
            # Layer 1: 384 -> (VSSLayer_up+upsample) -> 192
            # Layer 2: 192 -> (VSSLayer_up+upsample) -> 96
            # Layer 3: 96 -> (VSSLayer_up, no upsample) -> 96
            decoder_dims = [
                int(self.embed_dim * 2 ** (self.num_layers-2-i)) if i < self.num_layers-1 
                else self.embed_dim 
                for i in range(self.num_layers)
            ]
            
            print(f"[OK] Deep Supervision: decoder_dims = {decoder_dims}")
            
            for i, dim in enumerate(decoder_dims):
                # 每层一个1x1卷积作为辅助分割头
                aux_head = nn.Conv2d(
                    in_channels=dim,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    bias=False
                )
                self.aux_heads.append(aux_head)
            
            print(f"[OK] Deep Supervision enabled: {len(self.aux_heads)} auxiliary heads created")

        self.apply(self._init_weights)



    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _update_register(self, register_tokens, layer_features, layer_idx):
        """
        Register tokens 逐层更新机制（支持多种方案）
        
        实现"回收"机制：register tokens 逐层吸收全局上下文信息
        
        Args:
            register_tokens: (B, num_cls, C) - 当前的register状态
            layer_features: (B, H, W, C_layer) - 当前编码器层的输出特征
            layer_idx: int - 当前层索引
        
        Returns:
            updated_register: (B, num_cls, C) - 更新后的register tokens
        """
        B, num_cls, C = register_tokens.shape
        
        # === 方案A/V4：Cross-Attention（方案A和V4都使用）===
        if self.register_version in ['cross_attn', 'enhanced_gate']:
            # 1. 投影 layer_features 到 embed_dim 空间
            B_feat, H, W, C_layer = layer_features.shape
            layer_features_flat = layer_features.view(B_feat, H * W, C_layer)  # (B, H*W, C_layer)
            
            # 如果维度不匹配，需要投影
            if C_layer != self.embed_dim:
                layer_features_proj = self.register_projectors[layer_idx](layer_features_flat)  # (B, H*W, embed_dim)
            else:
                layer_features_proj = layer_features_flat
            
            # 2. Cross-Attention: Q from register, K/V from patches
            # register_tokens: (B, num_cls, embed_dim)
            # layer_features_proj: (B, H*W, embed_dim)
            attn_output = self.cross_attn_layers[layer_idx](
                query=register_tokens,  # (B, num_cls, embed_dim)
                key_value=layer_features_proj  # (B, H*W, embed_dim)
            )
            
            # 3. 残差连接 + 更新
            updated_register = register_tokens + attn_output
            
            return updated_register
        
        # === 方案D：Pooled Aggregation + Gated Update（备用）===
        elif self.register_version == 'pooled':
            # 1. 提取全局特征（通过自适应池化）
            layer_features_2d = layer_features.permute(0, 3, 1, 2)  # (B, H, W, C_layer) → (B, C_layer, H, W)
            global_feat = self.global_pool(layer_features_2d)  # (B, C_layer, 1, 1)
            global_feat = global_feat.view(B, -1)  # (B, C_layer)
            
            # 投影到统一维度
            global_feat = self.global_projectors[layer_idx](global_feat)  # (B, embed_dim)
            
            # 2. 拼接当前register状态和全局特征
            register_flat = register_tokens.view(B, num_cls * C)  # (B, num_cls * C)
            combined = torch.cat([register_flat, global_feat], dim=-1)  # (B, C * (num_cls + 1))
            
            # 3. 通过门控网络计算更新权重
            gate = self.register_updater(combined)  # (B, num_cls * C)
            gate = gate.view(B, num_cls, C)  # (B, num_cls, C)
            
            # 4. 门控更新：保留旧信息 + 吸收新信息
            global_feat_expanded = global_feat.unsqueeze(1).expand(B, num_cls, C)  # (B, num_cls, C)
            updated_register = register_tokens * (1 - gate) + global_feat_expanded * gate
            
            # 5. 应用层级scale
            if hasattr(self, 'layer_scales'):
                scale = self.layer_scales[layer_idx].view(1, 1, 1)
                updated_register = register_tokens + scale * (updated_register - register_tokens)
            
            return updated_register
        
        else:
            raise ValueError(f"Unknown register_version: {self.register_version}")

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, H, W, C)
        B, H, W, C = x.shape

        # === 方案D：初始化 register tokens ===
        register_tokens = None
        if self.use_register and self.num_cls_tokens > 0:
            # 初始化 register tokens（带位置编码）
            cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls  # (B, num_cls, C)
            register_tokens = cls_token  # 保持 3D 形状用于逐层更新
        
        # === 逐层处理编码器 + 逐层更新 register ===
        x_downsample = []
        for i, layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(x)  # Encoder layer 处理 patch tokens
            
            # 🔥 方案D核心：每层后更新 register tokens（回收全局信息）
            if register_tokens is not None:
                register_tokens = self._update_register(register_tokens, x, layer_idx=i)
        
        x = self.norm(x)  # B H W C

        # === 最终的 register_features 就是"回收"后的全局特征 ===
        if register_tokens is not None:
            register_features = register_tokens.view(B, -1)  # (B, num_cls * C)
        else:
            register_features = None

        return x, x_downsample, register_features

    # def forward_backbone(self, x):
    #     x = self.patch_embed(x)

    #     for layer in self.layers:
    #         x = layer(x)
    #     return x

    #Dencoder and Skip connection
    # def forward_up_features(self, x, x_downsample):
    #     for inx, layer_up in enumerate(self.layers_up):
    #         if inx == 0:
    #             x = layer_up(x)
    #         else:
    #             x = torch.cat([x,x_downsample[3-inx]],-1)
    #             x = self.concat_back_dim[inx](x)
    #             x = layer_up(x)
    #
    #     x = self.norm_up(x)  # B H W C
    #
    #     return x
    def forward_up_features(self, x, x_downsample, register_features=None):
        # === Deep Supervision: 保存每层特征 ===
        aux_outputs = []  # 存储辅助输出
        decoder_features = []  # 存储每层特征
        
        # === 新增：将 register 特征注入到解码器 ===
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            
            # === Deep Supervision: 保存当前层特征 ===
            if self.use_deep_supervision and self.training:
                decoder_features.append(x)  # (B, H, W, C)

        x = self.norm_up(x)  # B H W C

        # === 融合 register 全局上下文 ===
        if register_features is not None and self.use_register and hasattr(self, 'register_proj'):
            B, H, W, C = x.shape
            
            # 投影 register 特征
            register_proj = self.register_proj(register_features)  # (B, embed_dim)
            register_proj = register_proj.view(B, 1, 1, C).expand(B, H, W, C)  # 广播到所有空间位置
            
            # === 方案V4：双路径融合（全局+边界）===
            if self.register_version == 'enhanced_gate' and hasattr(self, 'edge_detector'):
                # 1. 提取边界特征
                x_2d = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
                boundary_feat = self.edge_detector(x_2d)  # (B, C, H, W)
                boundary_feat = boundary_feat.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
                
                # 2. 全局信息门控
                combined_global = torch.cat([x, register_proj], dim=-1)  # (B, H, W, 2*C)
                gate_global = self.global_gate(combined_global)  # (B, H, W, C)
                
                # 3. 边界信息门控
                combined_boundary = torch.cat([x, boundary_feat], dim=-1)  # (B, H, W, 2*C)
                gate_boundary = self.boundary_gate(combined_boundary)  # (B, H, W, C)
                
                # 4. 双路径融合（可学习的权重平衡）
                # 归一化融合权重
                w_global, w_boundary = torch.softmax(self.fusion_weights, dim=0)
                
                # 计算增强特征
                global_enhanced = register_proj * gate_global  # 全局增强
                boundary_enhanced = boundary_feat * gate_boundary  # 边界增强
                
                # 最终融合
                x = x + w_global * global_enhanced + w_boundary * boundary_enhanced
            
            # === 方案V2/V3：标准门控融合 ===
            elif hasattr(self, 'register_gate'):
                # 拼接局部特征和全局register特征
                combined = torch.cat([x, register_proj], dim=-1)  # (B, H, W, 2*C)
                
                # 通过门控网络计算自适应权重
                gate = self.register_gate(combined)  # (B, H, W, C)
                
                # 门控融合：gate接近1使用更多register，gate接近0使用更多原始特征
                x = x * (1 - gate) + register_proj * gate
            
            # === Fallback：简单加权 ===
            else:
                x = x + 0.1 * register_proj
        
        # === Deep Supervision: 生成辅助输出 ===
        if self.use_deep_supervision and self.training and hasattr(self, 'aux_heads'):
            # 获取目标尺寸（最终输出尺寸）
            B, H_final, W_final, C_final = x.shape
            target_size = (H_final * 4, W_final * 4)  # 上采样4倍到原始图像大小
            
            for i, feat in enumerate(decoder_features):
                # 转换为(B, C, H, W)格式
                feat_2d = feat.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
                
                # 通过辅助头生成预测
                aux_out = self.aux_heads[i](feat_2d)  # (B, num_classes, H, W)
                
                # 上采样到目标尺寸
                aux_out = F.interpolate(
                    aux_out,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                aux_outputs.append(aux_out)
            
            return x, aux_outputs
        
        return x

    def up_x4(self, x):
        if self.final_upsample=="expand_first":
            B,H,W,C = x.shape
            x = self.up(x)
            x = x.view(B, 4*H, 4*W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
            
        return x

    # def forward(self, x):
    #     x,x_downsample = self.forward_features(x)
    #     x = self.forward_up_features(x,x_downsample)
    #     x = self.up_x4(x)
    #     return x
    def forward(self, x):
        x, x_downsample, register_features = self.forward_features(x)  # 接收 register_features
        
        # === Deep Supervision: 处理辅助输出 ===
        if self.use_deep_supervision and self.training:
            decoder_out = self.forward_up_features(x, x_downsample, register_features)
            if isinstance(decoder_out, tuple):
                x, aux_outputs = decoder_out
                main_output = self.up_x4(x)
                return main_output, aux_outputs
        
        x = self.forward_up_features(x, x_downsample, register_features)  # 传递 register_features
        x = self.up_x4(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"


# APIs with VMamba2Dp =================
def check_vssm_equals_vmambadp():
    from bak.vmamba_bak1 import VMamba2Dp

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0


if __name__ == "__main__":
    # check_vssm_equals_vmambadp()
    model = VSSM().to('cuda')
    int = torch.randn(16,1,224,224).cuda()
    out = model(int)
    print(out.shape)
