# CrystaLLM - 钙钛矿生成指南（中文版）

[English Version](README.md) | 中文版

## 简介

CrystaLLM 是一个基于 Transformer 架构的大型语言模型，专门用于生成晶体结构的 CIF (Crystallographic Information File) 格式文件。本项目特别适合生成**稳定的钙钛矿结构**。

## 🚀 快速开始

### 一键运行演示

```bash
cd examples
./quickstart.sh
```

这个脚本会自动：
1. 检查并安装必要的依赖
2. 下载钙钛矿专用模型
3. 生成样例钙钛矿结构（CaTiO₃, SrTiO₃, BaTiO₃）
4. 分析并评估生成的结构
5. 生成详细的质量报告

### 批量生成钙钛矿

```bash
# 生成常见钙钛矿结构
python examples/generate_perovskites.py \
    --model crystallm_perov_5_small \
    --output perovskites/ \
    --num-samples 20

# 自定义组成和空间群
python examples/generate_perovskites.py \
    --model crystallm_perov_5_small \
    --output my_perovskites/ \
    --compositions "CaTiO3:Pm-3m,SrTiO3:Pm-3m,BaTiO3:P4mm" \
    --num-samples 50 \
    --temperature 0.7
```

### 分析生成的结构

```bash
# 全面分析和评估
python examples/analyze_perovskites.py \
    --input perovskites/ \
    --output analysis/

# 仅筛选高质量结构（≥80分）
python examples/analyze_perovskites.py \
    --input perovskites/ \
    --output analysis/ \
    --min-quality 80 \
    --valid-only
```

## 📚 详细文档

### 1. [钙钛矿生成完整指南](PEROVSKITE_GENERATION_GUIDE.md)

这是一份全面的中文指南，包含：

- **项目概述**：CrystaLLM 的核心原理和架构
- **代码分析**：详细解析各个模块的功能
- **生成方法**：
  - 方法一：使用预训练模型快速生成
  - 方法二：使用 MCTS 生成高质量结构
  - 方法三：训练专用钙钛矿模型
- **参数调优**：如何调整参数以获得更稳定的结构
- **实践案例**：真实的使用场景和示例代码
- **高级技术**：MCTS 解码和自定义评分函数

### 2. [示例脚本文档](examples/README.md)

包含两个主要的 Python 脚本：

#### `generate_perovskites.py` - 批量生成脚本
- 支持预设的常见钙钛矿列表
- 可自定义化学组成和空间群
- 自动后处理
- 生成详细报告

#### `analyze_perovskites.py` - 分析和筛选脚本
- 验证 CIF 文件有效性
- 计算 Goldschmidt 容忍因子
- 评估键长合理性
- 检查空间群一致性
- 综合质量评分（0-100分）
- 自动筛选高质量结构

## 🎯 核心功能

### 1. 质量评估指标

生成的结构会根据以下标准进行评分：

| 指标 | 分值 | 说明 |
|------|------|------|
| 结构有效性 | 40分 | CIF 格式正确，可构建有效晶体结构 |
| 空间群一致性 | 20分 | 生成的结构符合声明的空间群 |
| 原子位点重数一致性 | 15分 | 原子占据合理，重数与对称性匹配 |
| 键长合理性 | 15分 | 原子间距在合理范围内 |
| 容忍因子稳定性 | 10分 | Goldschmidt 容忍因子在 0.8-1.0 |

**总分 100 分**，建议选择得分 ≥ 80 分的结构。

### 2. 钙钛矿稳定性评估

对于 ABX₃ 型钙钛矿，脚本会自动计算 Goldschmidt 容忍因子：

```
t = (r_A + r_X) / (√2 × (r_B + r_X))
```

- **稳定** (0.8 ≤ t ≤ 1.0)：可能形成稳定的钙钛矿结构
- **边缘稳定** (0.7 ≤ t < 0.8)：可能稳定，需进一步验证
- **不稳定** (t < 0.7 或 t > 1.0)：不太可能形成钙钛矿结构

### 3. 支持的钙钛矿类型

#### 氧化物钙钛矿
- CaTiO₃, SrTiO₃, BaTiO₃ (钛酸盐)
- PbTiO₃, KNbO₃ (铁电材料)
- LaAlO₃ (基板材料)

#### 卤化物钙钛矿
- CsPbI₃, CsPbBr₃ (全无机钙钛矿太阳能电池)
- CsSnI₃ (无铅钙钛矿)
- MAPbI₃ (有机-无机杂化钙钛矿)

#### 双钙钛矿
- Sr₂FeMoO₆, Ba₂NiWO₆

## 💡 使用技巧

### 提高结构稳定性

1. **降低采样温度**：使用 `--temperature 0.6-0.7`
2. **减小 Top-K 值**：使用 `--top-k 5-8`
3. **增加样本数**：使用 `--num-samples 50-100`，然后筛选最佳结构
4. **指定空间群**：为已知的钙钛矿指定典型空间群

### 提高生成多样性

1. **提高采样温度**：使用 `--temperature 0.9-1.1`
2. **增大 Top-K 值**：使用 `--top-k 20-30`
3. **不指定空间群**：让模型自由探索

### 使用 MCTS 优化

对于重要的组成，建议使用 MCTS 方法：

```bash
# 创建提示文件
python bin/make_prompt_file.py CaTiO3 catio3_prompt.txt --spacegroup Pm-3m

# 使用 MCTS 生成
python bin/mcts.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:catio3_prompt.txt \
    num_simulations=300 \
    tree_width=10 \
    mcts_out_dir=catio3_mcts \
    device=cuda
```

## 📊 输出示例

### 生成报告示例
```
钙钛矿结构生成报告
============================================================

模型: crystallm_perov_5_small
生成参数:
  - 每种组成样本数: 20
  - 温度: 0.75
  - Top-K: 10
  - 设备: cuda

生成结果:
------------------------------------------------------------

CaTiO3:
  状态: ✓ 成功
  生成数量: 20
  处理后文件: results/perovskites/CaTiO3/processed/

SrTiO3:
  状态: ✓ 成功
  生成数量: 20
  处理后文件: results/perovskites/SrTiO3/processed/
```

### 分析报告示例
```
总体统计:
----------------------------------------------------------------------
总文件数: 60
成功分析: 58
分析失败: 2

结构有效性:
----------------------------------------------------------------------
有效结构: 54/60 (90.0%)
空间群一致: 56/60 (93.3%)
重数一致: 57/60 (95.0%)
平均键长合理性: 0.973

质量分数分布:
----------------------------------------------------------------------
平均质量分数: 78.3/100
优秀 (≥80分): 25 (41.7%)
良好 (60-79分): 28 (46.7%)
一般 (40-59分): 5 (8.3%)
较差 (<40分): 2 (3.3%)

Goldschmidt 容忍因子:
----------------------------------------------------------------------
平均值: 0.912
稳定性评估:
  稳定 (0.8-1.0): 48
  边缘稳定 (0.7-0.8): 6
  不稳定 (<0.7 or >1.0): 4
```

## 🔧 安装和依赖

### 基本依赖
```bash
pip install torch==2.0.1
pip install -r requirements.txt
pip install -e .
```

### 分析脚本额外依赖
```bash
pip install pymatgen pandas tqdm
```

## 📖 完整工作流程

```bash
# 1. 准备环境
cd /path/to/CrystaLLM
pip install -r requirements.txt
pip install -e .

# 2. 下载模型
python bin/download.py crystallm_perov_5_small.tar.gz
tar xzf crystallm_perov_5_small.tar.gz

# 3. 生成结构
python examples/generate_perovskites.py \
    --model crystallm_perov_5_small \
    --output results/perovskites \
    --num-samples 20

# 4. 分析结构
python examples/analyze_perovskites.py \
    --input results/perovskites \
    --output results/analysis

# 5. 查看结果
cat results/analysis/analysis_report.txt
cat results/analysis/high_quality_structures.csv
```

## 🤝 贡献

欢迎提交问题报告和改进建议！

## 📄 许可证

本项目遵循原 CrystaLLM 项目的许可证。

## 📚 引用

如果您在研究中使用了 CrystaLLM，请引用：

```bibtex
@article{antunes2024crystal,
  title={Crystal structure generation with autoregressive large language modeling},
  author={Antunes, Luis M and Butler, Keith T and Grau-Crespo, Ricardo},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={10761},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## 📞 联系方式

- **GitHub Issues**: 报告问题或提问
- **详细指南**: [PEROVSKITE_GENERATION_GUIDE.md](PEROVSKITE_GENERATION_GUIDE.md)
- **示例文档**: [examples/README.md](examples/README.md)

---

**祝您使用愉快！如果这些工具对您的研究有帮助，欢迎给项目一个 ⭐！**
