# 本机手写字符检测与识别

基于官方 Ultralytics `v8.4.175`（2026-10-09 发布），使用已公开的 YOLO26n。2026-10-12 核验时，YOLO27 的官方页面仍标记 Coming Soon，尚未公开实现和权重。本项目选择能实际安装、训练和验证的最新稳定模型。没有修改上游模型、训练循环或公共 Python 接口。

## 已收集数据

`sources.json` 固定了 27 个完整原始包的来源、大小和 SHA256。原始文件保留在本地，不包含在 Git 仓库中。HASYv2、CharLoc、CROHME 的校验值也匹配发布方 MD5；GNHK 使用公开镜像，校验值来自镜像 LFS，未与失效的官方二进制入口做字节比对。

| 数据                              | 在本项目中的实际用途                                                                           |
| --------------------------------- | ---------------------------------------------------------------------------------------------- |
| CASIA-HWDB2.0–2.2 + TGC-Diff 标注 | 恢复原页、匹配字符框、切为 512 像素检测图片；保留官方训练/测试写字人划分                       |
| CASIA-ICDAR2013                   | 完整留作后续独立测试，首轮不使用                                                               |
| CharLoc v3                        | 英文大小写与数字字符框；按 Writer ID 划分，排除 FilteredOut                                    |
| CASIA-HWDB1.0–1.2                 | 真实 GNT 字形组成多字符版式，标记为合成布局；训练和验证使用官方不同写字人                      |
| EMNIST                            | 从官方 train/test 读取真实字形，纠正存储方向后生成多字符版式；不累加重叠子集                   |
| HASYv2                            | 真实符号字形组成多字符版式；按 user_id 划分，避免复用交叉验证折造成重复                        |
| CROHME2023                        | 使用原始 InkML 轨迹和符号分组，同一坐标变换生成图像与字符框；首轮使用官方 train/val 的人工数据 |
| GNHK                              | 独立生成 `word_localization/data.yaml` 词定位任务，只选择 type=H；词框不混作字符框             |
| CASIA-HWDB-T                      | 保留字符串标签、接触点与原数据，供后续粘连分析；没有虚构字符矩形框，也不将其宣称为独立测试集   |

数据准备输出 `provenance.jsonl`，记录每张图片的来源、原始键、训练/验证归属、真实或合成版式及像素哈希。字符大小写、0/O、1/l/I 保留区别；只对明确列出的数学 LaTeX 别名进行映射。

## 下载、准备和训练

先安装带 CUDA 的 PyTorch，再在独立虚拟环境中安装此工作树：

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
python examples/handwriting/fetch.py --raw-root D:\Datasets\handwriting_raw
python examples/handwriting/prepare.py --raw-root D:\Datasets\handwriting_raw --output D:\Datasets\handwriting_yolo_pilot_clean --profile pilot
python examples/handwriting/train.py --data D:\Datasets\handwriting_yolo_pilot_clean\data.yaml --runs D:\Datasets\handwriting_yolo_runs --epochs 3 --batch 2
```

如果已经有本次整理好的原始数据，可使用 `fetch.py --verify-only` 检查源文件，不再下载和展开。`prepare.py` 要求空输出目录，避免旧标签或划分与新版本混用。

本机环境：RTX 5060 Laptop 8 GB，16 GB 内存；Python 3.14.6、PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130。YOLO26n 使用 512 输入、AMP、batch=2；上游训练内的验证批次会翻倍，batch=4 的初次运行在验证阶段显存不足，故当前配置为 2。关闭水平/垂直翻转、大角度旋转、Mosaic、MixUp，以免改变字符含义。训练使用已有 Ultralytics 引擎，随后从最佳权重重新运行验证。

整页推理使用重叠切片，把预测框移回原页并通过已有 torchvision NMS 去重，避免把大页直接缩小到 512 导致字符过小：

```powershell
python examples/handwriting/predict_page.py --model D:\Datasets\handwriting_yolo_runs\handwriting_pilot-2\weights\best.pt --image page.jpg --output page_result
```

结果为原页坐标的 JSON 和标注 JPG；输出是字符或原始数学符号组，不推断阅读顺序或公式结构。默认低置信阈值 0.05 用于观察首轮模型，不能把框的存在当成识别正确。

## 本机首轮实测

已完成 3 个训练轮次，并从保存的最佳权重重新独立验证。字符任务包含 2,709 张训练图片、604 张验证图片和 3,388 类；验证集有 280 类未出现在训练集中，训练集有 1,960 类不足 5 个实例。检测精确率 38.08%、召回率 1.23%、mAP50 1.39%、mAP50–95 1.12%。mAP 是检测指标，不是整页字符准确率；这个模型目前不能可靠识别手写页面。

独立整页 `C041-P19` 完全未参与训练：原始有效标签为 309 个字符，置信阈值 0.05 时预测 77 个框，其中类别正确且 IoU≥0.5 的只有 15 个，主要是标点。该页仅验证完整页面推理和坐标恢复流程，不代表整体 OCR 准确率。下一阶段需要提高每类样本量、扩大真实页面覆盖并延长训练；`pilot_metrics.json` 保存本次实测记录，数据和权重继续保留在 D 盘。

`--profile full` 扩大真实页面、词图和轨迹样本，并扩大字形组合池。它不是已经收敛的完整训练模型；首轮仅验证处理链、GPU 训练和初步指标。扩量前应增加稀有汉字覆盖、每类样本量与更多真实拍摄条件，再测独立测试集。

## 原始标注中的已知问题

- 中文字符框补充已匹配全部 5,391 页。`385-P20` 的一个框越界（x≈65535），转换时排除。
- 中文补充的两页残余文本不一致：`088-P16`、`007-P19`，首轮排除整页；原始文件保留。
- 空字符、不可打印字符和私用区无效标签被过滤，原始标签不修改。
- CharLoc 的一个框归一化高度为 `1.002810`，转换时排除。坐标十进制舍入误差只在 `1e-5` 范围内收回图像边界。
- 页面切片沿用字符类别，对边界框执行标准检测裁剪；切片属于派生数据，不替代原始页坐标。
- GNT、EMNIST、HASY 组合图的字符来源真实，布局为合成。它们的验证结果不能替代真实整页识别能力。
- CASIA 页面与 TGC-Diff 标注是同一批图像；HWDB-T 同样来自 CASIA，不当作额外独立样本统计。

## 数据许可

CASIA 限学术研究范围；CROHME 为 CC BY-NC-SA 3.0；HASYv2 的 Zenodo 元数据为 ODbL；GNHK 官方说明为 CC BY 4.0。CharLoc 存在声明差异：Zenodo 标 CC BY 4.0，压缩包 README 限非商业研究；获得作者澄清前按研究用途处理。代码许可证沿用上游 AGPL-3.0，不会覆盖数据自身许可。SCUT-HCCDoc、SCUT-EPT 和 IAM 的申请型数据没有纳入原始包。

## 为何有独立格式适配示例

上游现有转换器支持 COCO、DOTA、掩码与 NDJSON，未提供 GNT、DGRL、InkML 和上述手写标注格式。这里的新增代码只负责读取这些原始格式、坐标转换和数据来源记录；下载复用 `safe_download`，训练和验证复用官方引擎。放在 examples 下可以避免把研究数据的许可、二进制格式与取样规则混入通用模型库，也没有增加基础依赖。
