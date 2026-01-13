# SegMambaV1.0: Lightweight Semantic Segmentation with MobileMamba

SegMambaV1.0 is a high-performance semantic segmentation framework based on the **MobileMamba** backbone. It integrates the efficiency of State Space Models (SSM) with advanced wavelet transform techniques to achieve a superior balance between inference speed and segmentation accuracy.

## 🌟 Key Features

- **MobileMamba Backbone**: Leverages the lightweight Multi-Receptive Visual Mamba (MobileMamba) architecture, optimized for efficient feature extraction.
- **Wavelet-Enhanced Mamba (WTE-Mamba)**: Incorporates multi-level wavelet transforms to capture multi-scale features while maintaining computational efficiency.
- **MMSegmentation Integration**: Built upon the robust [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework, supporting standard training, evaluation, and deployment pipelines.
- **Medical & General Imaging**: Optimized for various datasets, including medical imaging (e.g., Kvasir-SEG) and general semantic segmentation tasks.
- **Patched SS2D**: Includes a patched version of the 2D Selective Scan (SS2D) for handling high-resolution images through sliding window processing.

## 🛠️ Installation

### Prerequisites
- Linux (Ubuntu 22.04 recommended)
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.6+ (for selective scan kernels)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShuiFengPu/SegMambaV1.0.git
   cd SegMambaV1.0
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Install MMSegmentation and its dependencies
   pip install -U openmim
   mim install mmengine
   mim install "mmcv>=2.0.0"
   ```

3. **Compile Selective Scan Kernels**:
   ```bash
   cd seg/backbones/lib_mamba/kernels/selective_scan
   python setup.py install
   ```

## 🚀 Getting Started

### Data Preparation
The project supports `.npy` format datasets. Organize your data as follows:
```text
data/kvasir_npy/
├── train_data/
├── train_label/
├── val_data/
└── val_label/
```
Update the `data_root` in `seg/configs/_base_/datasets/kvasir_npy.py` to point to your dataset path.

### Training
To train a model (e.g., U-Net with MobileMamba backbone):
```bash
python seg/tools/train.py seg/configs/U-Net/U-Net_MobileMamba.py
```

### Evaluation
To evaluate a trained model:
```bash
python seg/tools/test.py seg/configs/U-Net/U-Net_MobileMamba.py ${CHECKPOINT_PATH}
```

## 📊 Model Zoo

| Backbone | Head | Dataset | mIoU | Config |
| :--- | :--- | :--- | :--- | :--- |
| MobileMamba-B4 | U-Net | Kvasir-SEG | - | [config](seg/configs/U-Net/U-Net_MobileMamba.py) |
| MobileMamba-B4 | DeepLabV3 | ADE20K | - | [config](seg/configs/deeplabv3/deeplabv3_mobilemamba_b4-80k_ade20k-512x512.py) |

## 📂 Project Structure

- `seg/backbones/`: Implementation of MobileMamba and WTE-Mamba.
- `seg/configs/`: Configuration files for different models and datasets.
- `seg/my_mmseg/`: Custom dataset loaders and decode heads.
- `seg/tools/`: Scripts for training, testing, and analysis.

## 📜 Citation

If you find this project useful, please consider citing the original MobileMamba and SegMamba works:

```bibtex
@article{he2024mobilemamba,
  title={MobileMamba: Lightweight Multi-Receptive Visual Mamba Network},
  author={He, Haoran and others},
  journal={arXiv preprint arXiv:2411.15941},
  year={2024}
}

@article{xing2024segmamba,
  title={SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation},
  author={Xing, Zhaohuai and others},
  journal={arXiv preprint arXiv:2401.13560},
  year={2024}
}
```

## 🙏 Acknowledgements

This project is built upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and inspired by the [MobileMamba](https://github.com/lewandofskee/MobileMamba) and [SegMamba](https://github.com/ge-xing/SegMamba) repositories.
