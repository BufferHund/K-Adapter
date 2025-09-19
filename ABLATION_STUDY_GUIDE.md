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
    1.  **预训练**: 运行 `run_ablation_study_size.sh` 脚本。
    2.  **微调**: 为每个尺寸运行对应的 `run_finetune_single_adapter_size<SIZE>.sh` 脚本。
    3.  **数据收集**: 从微调结果文件 (`..._result.txt`) 中收集最终的F1分数。

### 实验一结果与分析

| Adapter 尺寸 | Micro F1 (综合性能) | Macro F1 (均衡性能) | Precision (精确率) | Recall (召回率) |
| :--- | :--- | :--- | :--- | :--- |
| 16 | **0.690** | 0.561 | 0.703 | 0.676 |
| 64 | 0.686 | 0.563 | 0.693 | **0.678** |
| 256 | **0.690** | **0.565** | 0.707 | 0.673 |
| 768 | 0.688 | 0.556 | **0.714** | 0.663 |

#### 核心发现

最关键的发现是，**Adapter的性能对尺寸变化不敏感**。在瓶颈层维度从16增加到768的过程中，所有关键性能指标都保持惊人地稳定，几乎没有变化。

#### 详细解读

1.  **高参数效率 (High Parameter Efficiency)**: 这个结果有力地证明了Adapter架构的有效性。一个尺寸仅为16的极小型适配器，其性能表现与一个尺寸大得多的768适配器相当。
2.  **任务特性推断**: 实验结果也暗示，对于OpenEntity这个实体分类任务，适配RoBERTa-large模型并不需要一个高维度的复杂适配器。

#### 结论

对于在OpenEntity任务上使用Factual Adapter，**明确推荐使用小尺寸的Adapter（例如16或64）**，因为它们在性能上与大尺寸Adapter几乎无差，但参数效率极高。

---

## 3. 实验二：Adapter 位置 (Adapter Position) 消融研究

- **目标**: 探究将Adapter插入在模型的不同位置（底层、中层、顶层）对下游任务性能的影响。
- **方法**:
    - **固定参数**: `adapter_size=64`
    - **可变参数**: `adapter_list` 在 `("0,1,2", "10,11,12", "21,22,23")` 中选择。
    - **基线**: 使用实验一中 `size=64`, `list="0,11,22"` 的结果作为“分散式”策略的基线。
- **执行步骤**:
    1.  **预训练**: 运行 `run_ablation_study_position.sh` 脚本（已修改为跳过基线）。
    2.  **微调**: 待预训练全部结束后，运行独立的微调脚本 `run_finetune_pos_early.sh`, `run_finetune_pos_middle.sh`, 和 `run_finetune_pos_late.sh`。
    3.  **数据收集**: 从每个微调任务的输出目录中收集F1分数，并与基线结果进行比较。

### 实验二结果与分析

实验运行完成后，我们从各个微调输出目录的 `_result.txt` 文件中，提取出最后一次评估在 `test` 测试集上的性能指标，汇总如下。

| 插入位置策略 | Adapter List | Micro F1 (测试集) | Macro F1 (测试集) |
| :--- | :--- | :--- | :--- |
| 底层 (Early) | `"0,1,2"` | 0.611 | 0.456 |
| **中层 (Middle)** | `"10,11,12"` | **0.706** | **0.585** |
| 分散式 (Baseline) | `"0,11,22"` | 0.686 | 0.563 |
| 顶层 (Late) | `"21,22,23"` | 0.389 | 0.371 |

#### 核心发现

与尺寸实验的平稳结果不同，**Adapter的插入位置对模型性能有决定性的影响**。将Adapter放置在模型的中层取得了最佳效果，而仅放置在顶层则导致性能严重下降。

#### 详细解读

1.  **中层是知识注入的“甜点区” (Sweet Spot)**: 将Adapter放置在模型的中部 (`"10,11,12"`) 取得了**最佳性能**，Micro F1达到了 **0.706**，甚至超越了我们之前所有实验的结果。这说明模型的中层部分，既包含了足够丰富的底层语言信息，又具备了初步的语义抽象能力，是进行任务适配的“最佳位置”。

2.  **顶层策略失败**: 最引人注目的是，将Adapter只放在模型的最后三层 (`"21,22,23"`)，其性能**急剧下降**至几乎随机猜测的水平。这强烈地暗示，对于OpenEntity任务，仅仅适配模型最高层的、最抽象的语义特征是远远不够的，甚至可能因为破坏了模型原有的能力而产生负面效果。

3.  **底层策略表现不佳**: 将Adapter只放在底层 (`"0,1,2"`) 的表现也不理想。这可能是因为模型的最底层主要处理的是词法、句法等基础语言特征，只在这里进行适配，无法有效地影响到模型后续更高层的语义决策过程。

4.  **分散式策略的有效性**: 我们之前的基线——分散式策略 (`"0,11,22"`)，其性能（0.686）非常稳健，证明了它是一种有效的通用策略。但本次实验也证明，针对特定任务，寻找最优的插入位置可以带来进一步的性能提升。

#### 结论

Adapter的插入位置至关重要。对于OpenEntity任务，RoBERTa-large的中层（10-12层附近）是进行知识注入和任务适配最有效率的位置。在未来的实验中，可以优先考虑使用**中层插入**的策略。

---

## 4. 实验三 (规划中)：Adapter 内部复杂度研究

- **目标**: 探究Adapter模块内部的微型Transformer层数 (`adapter_transformer_layers`) 所带来的影响。
- **方法**:
    - **固定参数**: `adapter_size=64`, `adapter_list="0,11,22"`。
    - **可变参数**: `adapter_transformer_layers`。
    - **基线**: 实验一中 `size=64` 的结果即为 `layers=2` 的基线。
    - **新实验组**:
        - `--adapter_transformer_layers 1` (更简单的Adapter)
        - `--adapter_transformer_layers 4` (更复杂的Adapter)
- **执行步骤**:
    1.  **预训练**: 运行 `run_ablation_study_layers.sh` 脚本。
    2.  **微调**: 运行对应的 `run_finetune_layers_1.sh` 和 `run_finetune_layers_4.sh` 脚本。
    3.  **分析**: 比较不同内部层数下的最终F1分数，观察增加内部复杂度是否能带来性能提升。

---

## 5. 实验四 (规划中)：与全量微调的效率对比

- **目标**: 量化对比 **Adapter-Tuning** 与 **Full Fine-tuning** 在 OpenEntity 任务上的性能表现和参数效率。
- **方法**:
    - **实验组1 (Adapter-Tuning)**: 复用实验一中 `size=16` 的结果。此方法冻结了主模型，只训练Adapter。
    - **实验组2 (Full Fine-tuning)**: 不加载任何Adapter，解冻并训练整个RoBERTa-large模型。
- **关键配置差异**:
    - **学习率**: 全量微调的学习率需要调小以防止破坏预训练权重，我们使用 `1e-5`。
    - **训练参数**: 全量微调时，`--freeze_bert` 参数需置空 (`""`)，且不加载任何Adapter (`--meta_fac_adaptermodel=""`)。
- **执行步骤**:
    1.  运行 `run_finetune_full.sh` 脚本来执行全量微调。
    2.  实验结束后，从输出目录中提取Micro F1分数。
    3.  计算并对比两种方法的可训练参数量和最终性能。
- **预期分析**:
    通过对比，可以清晰地展示出Adapter-Tuning用极小的参数量（预计不到全量微调的1%），达到了与全量微调相当的性能，从而有力地证明其参数效率的优越性。
