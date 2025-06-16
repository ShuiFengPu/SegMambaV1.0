import torch
import torch.nn.functional as F


def extract_sliding_patches(x, patch_size=64, stride=32):
    """
    提取输入张量的重叠滑动窗口patches

    Args:
        x (torch.Tensor): 输入张量，形状为 [B, C, H, W]
        patch_size (int): patch的大小，默认为64
        stride (int): 滑动步长，默认为32（50%重叠）

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


def reconstruct_from_patches(patches, patches_positions, original_size, stride=32):
    """
    从patches重建原始图像

    Args:
        patches (torch.Tensor): 形状为 [B, N, C, patch_size, patch_size] 的patches
        patches_positions (list): 每个patch在原图中的位置 [(h_start, h_end, w_start, w_end), ...]
        original_size (tuple): 原始图像大小 (H, W)
        stride (int): 滑动步长，默认为32

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


class PatchedSS2D(torch.nn.Module):
    """
    添加滑动窗口patch处理的SS2D包装类
    """

    def __init__(self, original_ss2d, patch_size=64, stride=32):
        """
        初始化PatchedSS2D

        Args:
            original_ss2d: 原始SS2D模块实例
            patch_size (int): patch的大小，默认为64
            stride (int): 滑动步长，默认为32（50%重叠）
        """
        super().__init__()
        self.original_ss2d = original_ss2d
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):
        """
        使用滑动窗口patch处理的forward方法

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            output (torch.Tensor): 处理后的输出，形状与输入相同
        """
        # 保存原始尺寸
        B, C, H, W = x.shape

        # 如果输入尺寸小于patch_size，直接使用原始forward
        if H <= self.patch_size and W <= self.patch_size:
            return self.original_ss2d(x)

        # 提取patches
        patches, patches_positions = extract_sliding_patches(x, self.patch_size, self.stride)

        # 重塑patches以批量处理
        B_orig, N, C, P_H, P_W = patches.shape
        patches_flat = patches.view(B_orig * N, C, P_H, P_W)

        # 应用原始SS2D到每个patch
        processed_patches = self.original_ss2d(patches_flat)

        # 恢复patches形状
        processed_patches = processed_patches.view(B_orig, N, C, P_H, P_W)

        # 重建输出
        output = reconstruct_from_patches(processed_patches, patches_positions, (H, W), self.stride)

        return output


def modify_mbwtconv2d_forward(self, x):
    """
    修改后的MBWTConv2d forward方法，在global_atten调用前添加滑动窗口处理

    Args:
        self: MBWTConv2d实例
        x (torch.Tensor): 输入张量

    Returns:
        output (torch.Tensor): 处理后的输出
    """
    # 原始forward逻辑的前半部分（直到global_atten调用前）
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

    # 在这里应用global_atten，它已经被包装为PatchedSS2D
    # 假设self.global_atten已经是PatchedSS2D的实例
    x = self.base_scale(self.global_atten(x))
    x = x + x_tag

    # 原始forward逻辑的后半部分
    if self.do_stride is not None:
        x = self.do_stride(x)

    return x


def patch_ss2d_modules(model, patch_size=64, stride=32):
    """
    递归地将模型中的所有SS2D模块替换为PatchedSS2D

    Args:
        model: PyTorch模型
        patch_size (int): patch的大小，默认为64
        stride (int): 滑动步长，默认为32（50%重叠）

    Returns:
        model: 修改后的模型
    """
    from backbones.lib_mamba.vmambanew import SS2D

    for name, module in list(model.named_children()):
        if isinstance(module, SS2D):
            # 替换为PatchedSS2D
            setattr(model, name, PatchedSS2D(module, patch_size, stride))
        else:
            # 递归处理子模块
            patch_ss2d_modules(module, patch_size, stride)

    return model


def test_patched_ss2d():
    """
    测试PatchedSS2D的功能
    """

    # 创建一个模拟的SS2D模块
    class MockSS2D(torch.nn.Module):
        def forward(self, x):
            # 简单地返回输入，用于测试
            return x

    # 创建PatchedSS2D
    original_ss2d = MockSS2D()
    patched_ss2d = PatchedSS2D(original_ss2d, patch_size=64, stride=32)

    # 创建测试输入
    batch_size = 2
    channels = 16
    height = 128
    width = 128
    x = torch.randn(batch_size, channels, height, width)

    # 使用PatchedSS2D处理
    output = patched_ss2d(x)

    # 验证输出形状
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"

    print("PatchedSS2D测试通过！")
    return output
