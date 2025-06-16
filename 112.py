# from PIL import Image
# import os

# mask_dir = '/tmp/SegMambaV1.0/seg/data/kvasir-seg-jpg/Kvasir-SEG/masks'
# for fname in os.listdir(mask_dir):
#     if fname.endswith('.jpg') or fname.endswith('.png'):
#         path = os.path.join(mask_dir, fname)
#         img = Image.open(path).convert('L')  # 👈 转为灰度图
#         img.save(path)  # 覆盖保存
import numpy as np
import os

# 替换成你的图像路径
img_path = '/tmp/SegMambaV1.0/seg/data/kvasir_npy/test_data_1.npy'
mask_path = '/tmp/SegMambaV1.0/seg/data/kvasir_npy/test_label_1.npy'

# 加载图像
img = np.load(img_path)
mask = np.load(mask_path)

print('--- 图像信息 ---')
print(f'图像 shape: {img.shape}')
print(f'图像 dtype: {img.dtype}')
print(f'图像 像素值范围: min={img.min()}, max={img.max()}')

print('\n--- 掩膜信息 ---')
print(f'掩膜 shape: {mask.shape}')
print(f'掩膜 dtype: {mask.dtype}')
print(f'掩膜 像素值唯一值: {np.unique(mask)}')
