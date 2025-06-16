import torch
import torch.nn as nn
import torch.nn.functional as F


def test_sliding_window_patch_integration():
    """
    测试滑动窗口patch集成的功能
    """

    # 创建一个模拟的cross_scan_fn函数
    def mock_cross_scan_fn(x):
        # 简单地返回输入，用于测试
        return x

    # 创建一个模拟的SS2D类
    class MockSS2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = 64
            self.stride = 32

        def _original_forward(self, x):
            # 模拟调用cross_scan_fn
            return mock_cross_scan_fn(x)

        def extract_sliding_patches(self, x, patch_size=64, stride=32):
            """
            提取输入张量的重叠滑动窗口patches
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

        def reconstruct_from_patches(self, patches, patches_positions, original_size, stride=32):
            """
            从patches重建原始图像
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

        def forward(self, x):
            """
            使用滑动窗口的forward实现
            """
            # 保存原始尺寸
            B, C, H, W = x.shape

            # 如果输入尺寸小于patch_size，直接使用原始forward
            if H <= self.patch_size and W <= self.patch_size:
                return self._original_forward(x)

            # 提取64×64的重叠patches
            patches, patches_positions = self.extract_sliding_patches(x, self.patch_size, self.stride)

            # 重塑patches以批量处理
            B_orig, N, C, P_H, P_W = patches.shape
            patches_flat = patches.view(B_orig * N, C, P_H, P_W)

            # 应用原始forward逻辑到每个patch
            processed_patches = self._original_forward(patches_flat)

            # 恢复patches形状
            processed_patches = processed_patches.view(B_orig, N, C, P_H, P_W)

            # 重建输出
            output = self.reconstruct_from_patches(processed_patches, patches_positions, (H, W), self.stride)

            return output

    # 创建测试输入
    batch_size = 2
    channels = 16
    height = 128
    width = 128
    x = torch.randn(batch_size, channels, height, width)

    # 创建模拟的SS2D实例
    ss2d = MockSS2D()

    # 使用滑动窗口forward处理
    output = ss2d(x)

    # 验证输出形状
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"

    # 测试不同尺寸的输入
    test_sizes = [(64, 64), (128, 64), (64, 128), (200, 150)]
    for h, w in test_sizes:
        x_test = torch.randn(batch_size, channels, h, w)
        output_test = ss2d(x_test)
        assert output_test.shape == x_test.shape, f"输入尺寸 {(h, w)} 的输出形状 {output_test.shape} 与输入形状 {x_test.shape} 不匹配"

    print("滑动窗口patch集成测试通过！")
    return output


def test_cross_scan_fn_compatibility():
    """
    测试cross_scan_fn与滑动窗口patch的兼容性

    注意：由于无法直接导入实际的cross_scan_fn，此测试使用模拟函数
    """

    # 创建一个模拟的cross_scan_fn函数
    def mock_cross_scan_fn(x):
        # 模拟cross_scan_fn的行为，添加一些变换以验证功能
        return x * 0.5 + 0.1

    # 创建测试输入
    batch_size = 2
    channels = 16
    height = 128
    width = 128
    x = torch.randn(batch_size, channels, height, width)

    # 直接应用mock_cross_scan_fn
    direct_output = mock_cross_scan_fn(x)

    # 使用滑动窗口patch处理
    # 提取patches
    patch_size = 64
    stride = 32

    # 计算需要的padding
    pad_h = (patch_size - height % stride) % stride if height % stride != 0 else 0
    pad_w = (patch_size - width % stride) % stride if width % stride != 0 else 0

    # 对输入进行padding
    if pad_h > 0 or pad_w > 0:
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, padded_height, padded_width = x_padded.shape
    else:
        x_padded = x
        padded_height, padded_width = height, width

    # 使用unfold提取patches
    patches = x_padded.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # 计算patches数量
    n_h = (padded_height - patch_size) // stride + 1
    n_w = (padded_width - patch_size) // stride + 1

    # 重塑张量形状
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size, n_h * n_w, channels, patch_size, patch_size)

    # 记录patch位置
    patches_positions = []
    for i in range(n_h):
        for j in range(n_w):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + patch_size
            w_end = w_start + patch_size
            patches_positions.append((h_start, h_end, w_start, w_end))

    # 重塑patches以批量处理
    patches_flat = patches.view(batch_size * n_h * n_w, channels, patch_size, patch_size)

    # 应用mock_cross_scan_fn到每个patch
    processed_patches = mock_cross_scan_fn(patches_flat)

    # 恢复patches形状
    processed_patches = processed_patches.view(batch_size, n_h * n_w, channels, patch_size, patch_size)

    # 重建输出
    patched_output = torch.zeros((batch_size, channels, padded_height, padded_width), device=x.device)
    count = torch.zeros((batch_size, 1, padded_height, padded_width), device=x.device)

    for i, (h_start, h_end, w_start, w_end) in enumerate(patches_positions):
        patched_output[:, :, h_start:h_end, w_start:w_end] += processed_patches[:, i]
        count[:, :, h_start:h_end, w_start:w_end] += 1

    patched_output = patched_output / (count + 1e-8)

    # 如果有padding，去除padding部分
    if pad_h > 0 or pad_w > 0:
        patched_output = patched_output[:, :, :height, :width]

    # 计算两种方法的差异
    diff = torch.abs(direct_output - patched_output).mean().item()
    print(f"直接处理与滑动窗口patch处理的平均差异: {diff:.6f}")

    # 由于重叠区域的平均处理，会有一些差异，但应该很小
    assert diff < 0.01, "直接处理与滑动窗口patch处理的差异过大"

    print("cross_scan_fn兼容性测试通过！")
    return direct_output, patched_output


def main():
    """
    运行所有测试
    """
    print("测试滑动窗口patch集成...")
    test_sliding_window_patch_integration()

    print("\n测试cross_scan_fn兼容性...")
    test_cross_scan_fn_compatibility()

    print("\n所有测试通过！")


if __name__ == "__main__":
    main()
