# K-Adapter 消融研究指南

本文档旨在详细记录对 K-Adapter 项目进行的一系列消融研究，包括对原始代码的关键修复、实验设计和执行步骤，以便于后续复现和理解。

## 1. 关键代码修复

在进行实验前，我们发现并修复了原始代码中的几个关键Bug。这些修复是成功运行实验的必要前提。

### 修复 1: 动态化 Adapter 内部配置

- **问题**: 在 `fac-adapter.py` 和 `examples/run_finetune_openentity_adapter.py` 中，`AdapterConfig` 类的 `hidden_size`, `intermediate_size`, 和 `num_attention_heads` 等参数被硬编码为固定值。这导致了两个问题：
    1.  当 `adapter_size` 不为768时，会因维度与注意力头数无法整除而崩溃。
    2.  当加载一个动态尺寸的预训练Adapter时，会因为模型结构与权重尺寸不匹配 (`size mismatch`) 而崩溃。

- **解决方案**: 我们将 `AdapterConfig` 修改为根据传入的 `adapter_size` 动态计算其内部参数，使其结构按比例缩放，更符合Transformer的设计原则。

    **修改前 (以`fac-adapter.py`为例):**
    ```python
    class AdapterConfig:
        # ...
        hidden_size: int=768
        intermediate_size: int=3072
        num_attention_heads: int=12
        # ...
    ```

    **修改后:**
    ```python
    class AdapterConfig:
        # ...
        hidden_size: int = self.adapter_size
        intermediate_size: int = self.adapter_size * 4
        num_attention_heads: int = max(1, self.adapter_size // 64)
        # ...
    ```
    *此修复已应用于 `fac-adapter.py` 和 `examples/run_finetune_openentity_adapter.py`。*

### 修复 2: 修复 `concat` 融合模式的逻辑错误

- **问题**: 在 `examples/run_finetune_openentity_adapter.py` 中，`ETModel` 的 `concat` 融合模式逻辑存在缺陷。它假设 `fac_adapter` 和 `lin_adapter` 总是同时存在，当只提供一个时，会因为试图使用一个未创建的变量 (`lin_adapter_outputs`) 而导致 `UnboundLocalError`。

- **解决方案**: 我们重写了 `concat` 部分的逻辑，使其变得健壮。现在它会检查每个Adapter的输出是否存在，然后再进行融合，无论是提供一个还是多个Adapter都能正常工作。

### 修复 3: 修复 `fac-adapter.py` 的最终模型保存逻辑

- **问题**: 在 `fac-adapter.py` 的 `main` 函数末尾，代码试图保存一个包含两个模型的元组 `(pretrained_model, adapter_model)`，这会导致 `AttributeError`。
- **解决方案**: 我们将其修正为只保存被训练的 `adapter_model`。

---

## 2. 实验一：Adapter 尺寸 (Adapter Size) 消融研究

- **目标**: 探究不同 `adapter_size` (瓶颈层维度) 对模型性能和参数量的影响。
- **方法**:
    - **固定参数**: `adapter_list="0,11,22"`
    - **可变参数**: `adapter_size` 在 `(768, 256, 64, 16)` 中选择。
- **执行步骤**:
    1.  **预训练**: 运行 `run_ablation_study_size.sh` 脚本。它会在 `ablation_study_output/fac_adapter_size_sub1e2` 目录下为每个尺寸预训练一个Factual Adapter。
    2.  **微调**:
        - 我们创建了一个模板脚本 `run_finetune_single_adapter_size16.sh`。
        - 通过复制和修改此模板，可以为每个尺寸创建一个微调脚本。
        - 运行这些脚本，它们会自动加载对应尺寸的预训练Adapter，并在 `outputs_light/` 目录下生成微调结果。
    3.  **数据收集**: 从预训练日志中收集可训练参数量，从微调结果文件 (`..._result.txt`) 中收集最终的F1分数，进行对比分析。

### 实验一结果与分析

实验运行完成后，我们从各个微调输出目录的 `_result.txt` 文件中，提取出最后一次评估在 `test` 测试集上的性能指标，汇总如下。

| Adapter 尺寸 | Micro F1 (综合性能) | Macro F1 (均衡性能) | Precision (精确率) | Recall (召回率) |
| :--- | :--- | :--- | :--- | :--- |
| 16 | **0.690** | 0.561 | 0.703 | 0.676 |
| 64 | 0.686 | 0.563 | 0.693 | **0.678** |
| 256 | **0.690** | **0.565** | 0.707 | 0.673 |
| 768 | 0.688 | 0.556 | **0.714** | 0.663 |

#### 核心发现

最关键的发现是，**Adapter的性能对尺寸变化不敏感**。在瓶颈层维度从16增加到768（参数量差异巨大）的过程中，所有关键性能指标（Micro F1, Macro F1, Precision, Recall）都保持惊人地稳定，几乎没有变化。

#### 详细解读

1.  **高参数效率 (High Parameter Efficiency)**: 这个结果有力地证明了Adapter架构的有效性。一个尺寸仅为16的极小型适配器，其性能表现与一个尺寸大得多的768适配器相当。这意味着我们可以用极小的参数量和计算开销，达到几乎无损的性能，这在资源受限或追求推理速度的场景下非常有价值。

2.  **任务特性推断**: 实验结果也暗示，对于OpenEntity这个实体分类任务，适配RoBERTa-large模型并不需要一个高维度的复杂适配器。任务所需的关键知识可能被成功地编码在了一个非常低维的子空间中。

#### 结论

对于在OpenEntity任务上使用Factual Adapter，**明确推荐使用小尺寸的Adapter（例如16或64）**。这可以在不牺牲模型性能的前提下，最大化地节约训练资源、降低模型存储开销。

---

## 3. 实验二：Adapter 位置 (Adapter Position) 消融研究

- **目标**: 探究将Adapter插入在模型的不同位置（底层、中层、顶层）对下游任务性能的影响。
- **方法**:
    - **固定参数**: `adapter_size=64`
    - **可变参数**: `adapter_list` 在 `("0,1,2", "10,11,12", "21,22,23")` 中选择。
    - **基线**: 使用实验一中 `size=64`, `list="0,11,22"` 的结果作为“分散式”策略的基线，无需重复运行。
- **执行步骤**:
    1.  **预训练**: 运行 `run_ablation_study_position.sh` 脚本。该脚本已修改为跳过基线，只运行3种新的位置配置。
    2.  **微调**: 待预训练全部结束后，运行独立的微调脚本 `run_finetune_pos_early.sh`, `run_finetune_pos_middle.sh`, 和 `run_finetune_pos_late.sh`。
    3.  **数据收集**: 从每个微调任务的输出目录中收集F1分数，并与基线结果进行比较，分析不同插入策略的优劣。