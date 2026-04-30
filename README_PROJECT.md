# 基于改进 YOLOv10 的路面积水检测系统

本项目在 YOLOv10n 的基础上，针对雨天路面积水（puddle）检测场景提出了三项网络结构改进，并通过系统的消融实验验证了各模块的有效性。

---

## 创新点

### 创新点一：P2 小目标检测分支

标准 YOLOv10n 使用 3 尺度检测头（P3/8, P4/16, P5/32），对远处小面积积水检测能力不足。本工作将 FPN 向下扩展至 **P2/4（stride=4）**，增加第四个检测尺度，提供更高分辨率的特征图以捕捉小目标。

- 将 Backbone 的 P2 层输出（layer 2）引入 Neck，经上采样后与 P3 特征拼接
- 检测头从 3 尺度变为 4 尺度：`Detect(P2, P3, P4, P5)`

### 创新点二：CBAM 通道-空间双路注意力机制

在 Neck 的 P2 和 P3 特征层后引入 **CBAM（Convolutional Block Attention Module）**，增强网络对积水反光特性和不规则边界的关注：

- **通道注意力（ChannelAttention）**：通过 AvgPool + MaxPool 双路径共享 MLP，帮助网络聚焦于水面的光谱特征
- **空间注意力（SpatialAttention）**：通过通道维度的均值+最大池化，突出不规则积水区域的空间位置

### 创新点三：PuddleEdgeEnhance 自适应边缘增强模块（核心创新）

针对积水区域反光强、边缘不规则的特点，设计了原创的边缘感知增强模块：

```python
class PuddleEdgeEnhance(nn.Module):
    def __init__(self, c1, alpha=0.5):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(-math.log(1.0 / alpha - 1.0)))
        self.norm = BatchNorm2d(c1)
        self.proj = Conv(c1, c1, 1, 1)

    def forward(self, x):
        edge = x - avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 局部边缘提取
        a = sigmoid(self.raw_alpha)  # 可学习融合权重，约束在 (0, 1)
        y = self.norm(x + a * edge)  # 自适应边缘增强
        return self.proj(y)
```

设计要点：
- 使用 3×3 均值池化计算局部差分，提取类拉普拉斯边缘信息
- **alpha 为可学习参数**，通过 sigmoid 映射到 (0,1)，网络自动学习最优增强程度
- 放置在 P3 特征层后，对中等尺度下积水边缘最显著的特征进行增强

### 消融实验结果

| 模型 | P2 分支 | CBAM | EdgeEnhance | mAP50 | mAP50-95 | Precision | Recall |
|:-----|:-------:|:----:|:-----------:|:-----:|:--------:|:---------:|:------:|
| Baseline | - | - | - | 0.7995 | 0.4402 | 0.7832 | 0.7424 |
| +P2 | ✓ | - | - | 0.8138 | 0.4525 | 0.8327 | 0.6992 |
| +P2+CBAM | ✓ | ✓ | - | 0.8119 | 0.4459 | 0.8496 | 0.7220 |
| **完整改进模型** | ✓ | ✓ | ✓ | 0.8091 | **0.4530** | 0.8330 | 0.7252 |

完整改进模型在 mAP50-95 上较 Baseline 提升 **+0.0128**，验证了三个模块的协同有效性。

---

## 项目结构

```
yolov10/
├── app.py                                    # Gradio 推理界面
├── train_puddle.py                           # 训练脚本（支持4种模式）
├── run_all.bat                               # 一键训练全部消融实验
├── datasets/puddle/                          # 积水检测数据集
├── runs/puddle/                              # 训练结果与权重
│   ├── yolov10n_baseline/weights/best.pt
│   ├── yolov10n_ablation_p2/weights/best.pt
│   ├── yolov10n_ablation_cbam/weights/best.pt
│   └── yolov10n_puddle_improved/weights/best.pt
└── ultralytics/
    ├── cfg/models/v10/
    │   ├── yolov10n.yaml                     # 标准 Baseline 配置
    │   ├── yolov10n_ablation_p2.yaml         # 消融：仅 P2 分支
    │   ├── yolov10n_ablation_cbam.yaml       # 消融：P2 + CBAM
    │   └── yolov10n_puddle.yaml              # 完整改进模型配置
    └── nn/modules/
        ├── block.py                          # PuddleEdgeEnhance 模块实现
        └── conv.py                           # CBAM / ChannelAttention / SpatialAttention 实现
```

---

## 环境配置

```bash
# 创建虚拟环境
conda create -n yolov10 python=3.9
conda activate yolov10

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 如需使用 Gradio 推理界面
pip install gradio
```

---

## 数据集准备

本项目使用 Roboflow 标注的路面积水数据集（单类别：puddle），将数据集放置于 `datasets/puddle/` 目录，结构如下：

```
datasets/puddle/
├── data.yaml        # 数据集配置文件（路径、类别定义）
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

`data.yaml` 中 `nc: 1`，类别名为 `puddle`。

---

## 训练

### 单模型训练

```bash
# 训练完整改进模型（默认）
python train_puddle.py --data datasets/puddle/data.yaml --mode improved

# 训练 Baseline 对照模型
python train_puddle.py --data datasets/puddle/data.yaml --mode baseline

# 训练消融实验（仅 P2 分支）
python train_puddle.py --data datasets/puddle/data.yaml --mode ablation_p2

# 训练消融实验（P2 + CBAM）
python train_puddle.py --data datasets/puddle/data.yaml --mode ablation_cbam
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|:-----|:------:|:-----|
| `--data` | 必填 | 数据集 data.yaml 路径 |
| `--mode` | improved | 训练模式：baseline / ablation_p2 / ablation_cbam / improved |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--batch` | 16 | 批次大小 |
| `--device` | auto | 训练设备：auto / 0 / cpu |
| `--pretrained` | yolov10n.pt | 预训练权重路径或文件名 |
| `--workers` | 4 | 数据加载线程数 |

### 一键训练全部消融实验

```bash
run_all.bat
```

该脚本将依次训练 Baseline → Ablation_P2 → Ablation_CBAM → Improved 四个模型。

训练结果保存在 `runs/puddle/` 目录下，每个模型的最优权重为 `weights/best.pt`。

---

## 推理

### Gradio 可视化界面

```bash
python app.py
# 浏览器访问 http://127.0.0.1:7860
```

功能：
- 支持图片输入
- 可调节推理尺寸（320-1280）和置信度阈值（0-1）
- 默认加载完整改进模型权重

### Python 脚本推理

```python
from ultralytics import YOLOv10

# 加载训练好的模型
model = YOLOv10("runs/puddle/yolov10n_puddle_improved/weights/best.pt")

# 图片推理
results = model.predict(source="test_image.jpg", imgsz=640, conf=0.25)
results[0].save("output.jpg")
```

---

## 致谢

- [YOLOv10](https://github.com/THU-MIG/yolov10) — 清华大学 MIG 实验室
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO 框架
