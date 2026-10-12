# AllFreqNet 中文使用说明

这是从 `version3` 整理的 GitHub 代码版本，对应论文
*AllFreqNet: A Unified Frequency-Enhanced All-in-One Framework for Medical Image Restoration*。
支持 CT 去噪、CT 金属伪影去除、MRI 超分辨率和 PET 合成。

模型和主要模块已按论文统一命名：`AllFreqNet`、`MSMoE` 和 RC-FPM 相关组件，
分别位于 `model/allfreqnet.py`、`model/ms_moe.py` 和 `model/rc_fpm.py`。
通过 `from model import AllFreqNet` 使用模型；`AllFreqNet` 现在是真实类名。
原参数属性名保留，以兼容已有权重，网络计算过程没有修改。
完整的新旧名称对应关系见 [docs/NAMING.md](docs/NAMING.md)。

## 安装与数据

先按照 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装适合自己
CPU/CUDA 的 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
```

使用 DICOM 时改为安装 `requirements-dicom.txt`。建议新环境采用 Python 3.10
或更新版本；本次也在原机 Python 3.7 / PyTorch 1.13.1 CPU 环境做了运行验证。

数据放在仓库外，或者被 Git 忽略的 `datasets/All-in-One/`。每个任务使用：

```text
CT/train/LQ/                 CT/train/HQ/
CT/validation/LQ/            CT/validation/HQ/
CT/test/LQ/                  CT/test/HQ/
```

其他任务将 `CT` 替换为 `MA`、`PET` 或 `MRI`。LQ/HQ 文件名和图像尺寸必须一致。
代码处理二维切片，三维体数据需要先切片。格式、强度范围和数据划分说明见
[docs/DATA.md](docs/DATA.md)。权重与真实数据没有装进仓库。

## 运行

在仓库根目录执行；下面是一行式命令，PowerShell 可以直接使用。路径有空格时加引号。

训练：

```bash
python train.py --data-root datasets/All-in-One --modalities CT MA PET MRI --output-dir runs/AllFreqNet --device cuda
```

评测：

```bash
python test.py --data-root datasets/All-in-One --weights weights/Generator_iteration_199000.pth --output-dir results/test --save-images --device cuda
```

无 HQ 的推理：

```bash
python infer.py --data-root datasets/All-in-One --weights weights/Generator_iteration_199000.pth --modalities CT --output-dir results/inference --output-format nii.gz --device cuda
```

没有 CUDA 时使用 `--device cpu`。RAW 默认是 512×512 的小端 float32，其他尺寸
用 `--raw-shape 高 宽` 指定。运行入口均支持 `--help`。

原目录的 `weight/Generator_iteration_199000.pth` 可自行复制到新目录的 `weights/`，
或直接用 `--weights` 指定它的位置。权重已做严格参数匹配及清理前后输出一致性检查。
训练结果在 `runs/`，评测及推理结果在 `results/`，均由 `.gitignore` 排除。

## 已验证与待补充内容

本次完成 9 个回归测试，并用合成数据跑通两次训练迭代、四任务评测及无 HQ 推理。
这不等于重新复现论文中的实验指标；真实数据集和 CUDA 完整训练尚未运行。

原实现有一些需要作者核实的论文对应关系，例如第一层 MoE 的输出未进入后续编码器，
以及专家合并使用对数域计算。本版本保留这些行为，详情见
[docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)。

公开的权重下载地址、原始数据划分清单和最终发表信息尚未提供，文档没有虚构这些内容。
若要对外承诺完整复现论文结果，应补齐对应资料。

许可证沿用原目录的 Apache-2.0，同时保留 AMIR、Restormer 和 SSIM 来源说明。
完整英文说明和引用格式见 [README.md](README.md)。
