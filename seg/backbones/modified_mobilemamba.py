import torch
import itertools
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from backbones.lib_mamba.vmambanew import SS2D
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
from timm.layers import DropPath
from torch.nn.modules.batchnorm import _BatchNorm
from mmseg.registry import MODELS


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def extract_sliding_patches(x, patch_size=32, stride=16):
    """
    提取输入张量的重叠滑动窗口patches

    Args:
        x (torch.Tensor): 输入张量，形状为 [B, C, H, W]
        patch_size (int): patch的大小，默认为32
        stride (int): 滑动步长，默认为16（50%重叠）

    Returns:
        patches (torch.Tensor): 提取的patches，形状为 [B, N, C, patch_size, patch_size]
                               其中N是patches的数量
        patches_positions (list): 每个patch在原图中的位置 [(h_start, h_end, w_start, w_end), ...]
    """
    B, C, H, W = x.shape

    # 计算需要的padding
    pad_h = (patch_size - H % stride) % stride if H % stride != 0 else 0
    pad_w = (patch_size - W % stride) % stride if W % stride != 0 else 0

    # 对输入进行padding，确保能够完整提取patches
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        B, C, H, W = x.shape

    # 使用unfold提取patches
    patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # 计算patches数量
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1

    # 重塑张量形状为 [B, n_h*n_w, C, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, n_h * n_w, C, patch_size, patch_size)

    # 记录每个patch的位置
    patches_positions = []
    for i in range(n_h):
        for j in range(n_w):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + patch_size
            w_end = w_start + patch_size
            patches_positions.append((h_start, h_end, w_start, w_end))

    return patches, patches_positions


def reconstruct_from_patches(patches, patches_positions, original_size, stride=16):
    """
    从patches重建原始图像

    Args:
        patches (torch.Tensor): 形状为 [B, N, C, patch_size, patch_size] 的patches
        patches_positions (list): 每个patch在原图中的位置 [(h_start, h_end, w_start, w_end), ...]
        original_size (tuple): 原始图像大小 (H, W)
        stride (int): 滑动步长，默认为16

    Returns:
        output (torch.Tensor): 重建的图像，形状为 [B, C, H, W]
    """
    B, N, C, patch_size, _ = patches.shape
    H, W = original_size

    # 计算需要的padding
    pad_h = (patch_size - H % stride) % stride if H % stride != 0 else 0
    pad_w = (patch_size - W % stride) % stride if W % stride != 0 else 0

    # 创建输出张量和计数张量（用于平均重叠区域）
    output = torch.zeros((B, C, H + pad_h, W + pad_w), device=patches.device)
    count = torch.zeros((B, 1, H + pad_h, W + pad_w), device=patches.device)

    # 将patches放回原位置
    for i, (h_start, h_end, w_start, w_end) in enumerate(patches_positions):
        output[:, :, h_start:h_end, w_start:w_end] += patches[:, i]
        count[:, :, h_start:h_end, w_start:w_end] += 1

    # 对重叠区域取平均
    output = output / (count + 1e-8)

    # 如果有padding，去除padding部分
    if pad_h > 0 or pad_w > 0:
        output = output[:, :, :H, :W]

    return output


class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',
                 ssm_ratio=1, forward_type="v05"):
        super(MBWTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 保存原始的global_atten实现
        self.global_atten_original = SS2D(d_model=in_channels, d_state=1, ssm_ratio=ssm_ratio,
                                          initialize="v2", forward_type=forward_type,
                                          channel_first=True, k_group=2)

        # 创建一个新的global_atten方法，它将使用滑动窗口
        self.global_atten = self._patched_global_atten

        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def _patched_global_atten(self, x):
        """
        使用滑动窗口的global_atten实现
        """
        # 保存原始尺寸
        B, C, H, W = x.shape

        # 提取32x32的重叠patches
        patches, patches_positions = extract_sliding_patches(x, patch_size=32, stride=16)

        # 重塑patches以批量处理
        B_orig, N, C, P_H, P_W = patches.shape
        patches_flat = patches.view(B_orig * N, C, P_H, P_W)

        # 应用原始global_atten到每个patch
        processed_patches = self.global_atten_original(patches_flat)

        # 恢复patches形状
        processed_patches = processed_patches.view(B_orig, N, C, P_H, P_W)

        # 重建输出
        output = reconstruct_from_patches(processed_patches, patches_positions, (H, W), stride=16)

        return output

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # 使用修改后的global_atten（已经包含滑动窗口处理）
        x = self.base_scale(self.global_atten(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

# 其他类的定义保持不变...
class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple


class MobileMambaModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d", ):
        super().__init__()
        # 输入特征通道维度
        self.dim = dim

        # 计算全局通道数（调整为16的最近倍数，优化计算效率）
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))

        # 动态分配局部通道数（保证全局+局部通道不超总维度）
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels  # 剩余通道给局部
        else:
            self.local_channels = int(local_ratio * dim)  # 按比例分配

        # 剩余通道作为identity直连通道（类似残差连接）
        self.identity_channels = self.dim - self.global_channels - self.local_channels

        # 局部特征处理分支（深度可分离卷积）
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(  # 含BN和ReLU的深度卷积
                self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()  # 无操作分支

        # 全局特征处理分支（含SSM的改进卷积）
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(  # 结合状态空间模型的卷积
                self.global_channels, self.global_channels, kernels,
                wt_levels=1, ssm_ratio=ssm_ratio,
                forward_type=forward_type, )
        else:
            self.global_op = nn.Identity()

        # 特征融合投影层（整合多分支结果）
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),  # 激活函数
            Conv2d_BN(dim, dim, bn_weight_init=0, )  # 1x1卷积+BN（通道数不变）
        )

    def forward(self, x):  # x (B,C,H,W)
        # 将特征通道切分为全局/局部/直连三部分
        x1, x2, x3 = torch.split(
            x,
            [self.global_channels, self.local_channels, self.identity_channels],
            dim=1
        )

        # 分支处理（保留原始维度）
        x1 = self.global_op(x1)  # 全局特征提取（大感受野）
        x2 = self.local_op(x2)  # 局部特征提取（细节纹理）
        # x3作为identity直连通道（不处理）

        # 通道拼接 + 投影融合
        x = self.proj(torch.cat([x1, x2, x3], dim=1))  # 维度重组与非线性增强
        return x

class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d",):
        super().__init__()

        self.dim = dim
        self.attn = MobileMambaModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                           kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type,)

    def forward(self, x):
        x = self.attn(x)
        return x

class MobileMambaBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(MobileMambaBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio,
                                                       kernels=kernels, ssm_ratio=ssm_ratio,forward_type=forward_type))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


# 注册模型到自定义的模型仓库（通过装饰器实现）
@MODELS.register_module()
class MobileMamba(torch.nn.Module):
    def __init__(self,
                 img_size=224,  # 输入图像尺寸
                 in_chans=3,  # 输入通道数（RGB图像为3）
                 num_classes=1000,  # 分类类别数（ImageNet默认）
                 stages=['s', 's', 's'],  # 阶段类型标记（'s'表示标准块）
                 embed_dim=[192, 384, 448],  # 各阶段特征嵌入维度
                 global_ratio=[0.8, 0.7, 0.6],  # 全局特征占比（WTE-Mamba部分）
                 local_ratio=[0.2, 0.2, 0.3],  # 局部特征占比（MK-DeConv部分）
                 depth=[1, 2, 2],  # 各阶段块重复次数
                 kernels=[7, 5, 3],  # 各阶段卷积核尺寸（多核设计）
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],  # 下采样操作配置
                 distillation=False,  # 是否启用蒸馏训练
                 drop_path=0,  # DropPath的丢弃概率
                 ssm_ratio=1,  # 状态空间模型通道缩放比
                 forward_type="v052d",  # Mamba前向计算版本
                 sync_bn=False,  # 是否同步跨GPU的BatchNorm
                 out_indices=(1, 2, 3),  # 输出特征图的阶段索引
                 pretrained=None,  # 预训练权重路径
                 frozen_stages=-1,  # 冻结训练的阶段数（-1表示不冻结）
                 norm_eval=False):  # 是否在训练时固定BN统计量
        super().__init__()

        # 同步BatchNorm配置
        self.sync_bn = sync_bn
        # 指定输出特征图的阶段
        self.out_indices = out_indices
        # 预训练权重路径
        self.pretrained = pretrained
        # 冻结训练的阶段数
        self.frozen_stages = frozen_stages
        # BN层评估模式设置
        self.norm_eval = norm_eval

        resolution = img_size  # 保留原始分辨率参数

        # 构建Patch Embedding模块（4次下采样）
        self.patch_embed = torch.nn.Sequential(
            # 第1次下采样（3->24通道，stride=2）
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1),
            torch.nn.ReLU(),  # 激活函数
            # 第2次下采样（24->48通道，stride=2）
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1),
            # 第3次下采样（48->96通道，stride=2）
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1),
            # 第4次下采样（96->192通道，stride=2）
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
        )

        # 初始化三个阶段块列表
        self.blocks1 = []  # 阶段1
        self.blocks2 = []  # 阶段2
        self.blocks3 = []  # 阶段3

        # 生成DropPath概率的线性空间（用于随机深度）
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        # 循环构建三个阶段的核心块
        for i, (stg, ed, dpth, gr, lr, do) in enumerate(
                zip(stages, embed_dim, depth, global_ratio, local_ratio, down_ops)):
            # 获取当前阶段的DropPath概率切片
            dpr = dprs[sum(depth[:i]):sum(depth[:i + 1])]

            # 构建当前阶段的多个块
            for d in range(dpth):
                # 动态获取块列表（blocks1/blocks2/blocks3）
                block_list = eval('self.blocks' + str(i + 1))
                # 添加MobileMambaBlock（核心模块）
                block_list.append(
                    MobileMambaBlock(
                        stg,  # 阶段类型
                        ed,  # 嵌入维度
                        gr,  # 全局特征比例
                        lr,  # 局部特征比例
                        kernels[i],  # 卷积核尺寸
                        dpr[d],  # 当前块的DropPath概率
                        ssm_ratio=ssm_ratio,
                        forward_type=forward_type
                    )
                )

            # 处理下采样操作
            if do[0] == 'subsample':
                # 获取下一阶段的块列表（blocks2/blocks3）
                blk = eval('self.blocks' + str(i + 2))
                # 添加残差连接+下采样模块
                blk.append(
                    torch.nn.Sequential(
                        # 残差分支1：分组卷积
                        Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                        # 残差分支2：FFN扩展
                        Residual(FFN(embed_dim[i], int(embed_dim[i] * 2)))
                    )
                )
                # 添加Patch Merging（特征图下采样）
                blk.append(PatchMerging(*embed_dim[i:i + 2]))
                # 添加下采样后的残差模块
                blk.append(
                    torch.nn.Sequential(
                        Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1])),
                        Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2)))
                    )
                )

        # 将列表转换为Sequential模块
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        # 初始化权重
        self._init_weights()
        # 同步BN处理
        self._sync_bn() if sync_bn else None
        # 冻结指定阶段
        self._freeze_stages()

    # 权重初始化方法
    def _init_weights(self):
        # 无预训练权重时的初始化
        if self.pretrained is None:
            for m in self.parameters():
                # 线性层使用截断正态分布初始化
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                # LayerNorm层初始化
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        # 加载预训练权重
        else:
            state_dict = torch.load(self.pretrained, map_location='cpu')
            self_state_dict = self.state_dict()
            # 权重对齐加载
            for k, v in state_dict.items():
                if k in self_state_dict.keys():
                    self_state_dict.update({k: v})
            self.load_state_dict(self_state_dict, strict=True)
            print(f'load ckpt from {self.pretrained}')

    # 同步跨GPU的BatchNorm统计量
    def _sync_bn(self):
        self.patch_embed = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.patch_embed)
        self.blocks1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.blocks1)
        self.blocks2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.blocks2)
        self.blocks3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.blocks3)

    # 忽略某些参数的权重衰减（如LN的bias）
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'token'}

    # 忽略指定关键字的权重衰减
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'alpha', 'gamma', 'beta'}

    # 定义不进行微调的参数
    @torch.jit.ignore
    def no_ft_keywords(self):
        return { }

    # 定义需要微调的头部参数
    @torch.jit.ignore
    def ft_head_keywords(self):
        return {'head.weight', 'head.bias'}, self.num_classes

    # 获取分类器
    def get_classifier(self):
        return self.head

    # 重置分类器（适配迁移学习）
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()

    # 修复BN层的统计量（防止NaN）
    def check_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
                m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)

    # 前向传播过程
    def forward(self, x):
        out = []  # 存储各阶段输出
        # 初始下采样（224x224 -> 14x14）
        x = self.patch_embed(x)
        out.append(x)  # 阶段0输出
        # 阶段1处理
        x = self.blocks1(x)
        out.append(x)  # 阶段1输出
        # 阶段2处理
        x = self.blocks2(x)
        out.append(x)  # 阶段2输出
        # 阶段3处理
        x = self.blocks3(x)
        out.append(x)  # 阶段3输出
        # 返回指定阶段的输出
        return tuple([out[i] for i in self.out_indices])

    # 冻结指定阶段的参数
    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            m = getattr(self, f'blocks{i}')  # 获取阶段模块
            m.eval()  # 设置为评估模式
            for param in m.parameters():
                param.requires_grad = False  # 冻结梯度

    # 训练模式设置（保持BN层冻结）
    def train(self, mode=True):
        super(MobileMamba, self).train(mode)
        self._freeze_stages()  # 维持冻结状态
        if mode and self.norm_eval:  # 训练模式下固定BN统计量
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

