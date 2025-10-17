# 🔬 CrystaLLM 钙钛矿生成工具包

## 快速链接

| 资源 | 描述 | 行数 |
|------|------|------|
| [快速开始](#快速开始) | 一键运行演示 | - |
| [完整指南](PEROVSKITE_GENERATION_GUIDE.md) | 详细的中文指南 | 724 行 |
| [中文 README](README_CN.md) | 快速入门文档 | 304 行 |
| [示例脚本](examples/) | 实用工具脚本 | 866 行 |
| [更新日志](CHANGELOG_PEROVSKITE.md) | 新增功能说明 | 121 行 |

---

## 快速开始

### 🚀 一键运行（推荐新手）

```bash
cd examples
./quickstart.sh
```

这将自动完成：
1. ✓ 检查并安装依赖
2. ✓ 下载钙钛矿模型
3. ✓ 生成样例结构（CaTiO₃, SrTiO₃, BaTiO₃）
4. ✓ 分析和评估
5. ✓ 生成质量报告

### 📝 手动运行

```bash
# 1. 批量生成钙钛矿
python examples/generate_perovskites.py \
    --model crystallm_perov_5_small \
    --output perovskites/ \
    --num-samples 20

# 2. 分析生成的结构
python examples/analyze_perovskites.py \
    --input perovskites/ \
    --output analysis/ \
    --min-quality 80
```

---

## 🎯 主要功能

### 1. 批量生成（generate_perovskites.py）
- ✅ 预设 12+ 种常见钙钛矿
- ✅ 支持自定义组成和空间群
- ✅ 自动创建提示和后处理
- ✅ 生成详细的任务报告

### 2. 智能分析（analyze_perovskites.py）
- ✅ 多维度质量评估（100 分制）
- ✅ Goldschmidt 容忍因子计算
- ✅ 键长合理性检查
- ✅ 自动筛选高质量结构

### 3. 一键演示（quickstart.sh）
- ✅ 全自动化流程
- ✅ 依赖检查和安装
- ✅ 友好的用户提示

---

## 📚 文档导航

### 🌟 推荐阅读顺序

1. **初学者**：
   ```
   README_CN.md → 运行 quickstart.sh → examples/README.md
   ```

2. **进阶用户**：
   ```
   PEROVSKITE_GENERATION_GUIDE.md → 自定义生成 → 参数调优
   ```

3. **研究人员**：
   ```
   PEROVSKITE_GENERATION_GUIDE.md（MCTS 章节） → 自定义评分函数
   ```

### 📖 详细文档

#### [PEROVSKITE_GENERATION_GUIDE.md](PEROVSKITE_GENERATION_GUIDE.md)
**724 行完整中文指南**

- 第 1 章：项目概述
- 第 2 章：核心原理（模型、CIF、钙钛矿）
- 第 3 章：代码架构分析（5 个核心模块）
- 第 4 章：生成方法（3 种完整流程）
- 第 5 章：参数调优
- 第 6 章：实践案例
- 第 7 章：高级技术（MCTS、自定义评分）

#### [README_CN.md](README_CN.md)
**304 行快速入门**

- 一键开始命令
- 质量评估指标
- 支持的钙钛矿类型
- 使用技巧
- 完整工作流程
- 输出示例

#### [examples/README.md](examples/README.md)
**383 行脚本文档**

- 脚本功能说明
- 参数列表
- 使用示例
- 高级用法
- 故障排除

---

## 🔬 支持的钙钛矿类型

### 氧化物钙钛矿
- **钛酸盐**：CaTiO₃, SrTiO₃, BaTiO₃
- **铁电材料**：PbTiO₃, KNbO₃
- **基板材料**：LaAlO₃

### 卤化物钙钛矿（太阳能电池）
- **全无机**：CsPbI₃, CsPbBr₃
- **无铅**：CsSnI₃
- **杂化**：MAPbI₃

### 双钙钛矿
- **磁性材料**：Sr₂FeMoO₆
- **介电材料**：Ba₂NiWO₆

---

## 📊 质量评估系统

### 综合评分（0-100 分）

| 指标 | 分值 | 说明 |
|------|------|------|
| 结构有效性 | 40 | CIF 格式正确，可构建晶体结构 |
| 空间群一致性 | 20 | 符合声明的空间群 |
| 原子位点重数一致性 | 15 | 原子占据合理 |
| 键长合理性 | 15 | 原子间距正常 |
| 容忍因子稳定性 | 10 | Goldschmidt t 在 0.8-1.0 |

### 容忍因子评估

```
t = (r_A + r_X) / (√2 × (r_B + r_X))
```

- **稳定**：0.8 ≤ t ≤ 1.0
- **边缘稳定**：0.7 ≤ t < 0.8
- **不稳定**：其他

---

## 💡 使用技巧

### 提高结构稳定性
```bash
python examples/generate_perovskites.py \
    --model crystallm_perov_5_small \
    --temperature 0.6 \    # 降低温度
    --top-k 5 \            # 减小 Top-K
    --num-samples 50       # 增加样本数
```

### 提高生成多样性
```bash
python examples/generate_perovskites.py \
    --model crystallm_perov_5_small \
    --temperature 1.0 \    # 提高温度
    --top-k 30 \           # 增大 Top-K
    --num-samples 20       # 中等样本数
```

### 使用 MCTS 优化
```bash
python bin/mcts.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:prompt.txt \
    num_simulations=300 \  # 增加模拟次数
    tree_width=10          # 控制搜索宽度
```

---

## 📈 典型输出

### 生成报告示例
```
钙钛矿结构生成报告
============================================================
模型: crystallm_perov_5_small
生成参数: temperature=0.75, top_k=10, num_samples=20

CaTiO3: ✓ 成功 (20 个结构)
SrTiO3: ✓ 成功 (20 个结构)
BaTiO3: ✓ 成功 (20 个结构)
```

### 分析报告示例
```
总体统计:
  总文件数: 60
  有效结构: 54/60 (90.0%)
  
质量分数分布:
  平均质量分数: 78.3/100
  优秀 (≥80分): 25 (41.7%)
  良好 (60-79分): 28 (46.7%)
  
容忍因子:
  平均值: 0.912
  稳定结构: 48
```

---

## 🔧 安装和依赖

### 基本依赖
```bash
# PyTorch
pip install torch==2.0.1

# CrystaLLM
pip install -r requirements.txt
pip install -e .

# 分析工具
pip install pymatgen pandas tqdm
```

### 下载模型
```bash
# 钙钛矿专用小模型（推荐）
python bin/download.py crystallm_perov_5_small.tar.gz
tar xzf crystallm_perov_5_small.tar.gz

# 钙钛矿专用大模型（更高质量）
python bin/download.py crystallm_perov_5_large.tar.gz
tar xzf crystallm_perov_5_large.tar.gz
```

---

## 🎓 学习路径

### 第 1 天：入门
1. 阅读 [README_CN.md](README_CN.md)
2. 运行 `./examples/quickstart.sh`
3. 查看生成的结构和报告

### 第 2 天：实践
1. 阅读 [examples/README.md](examples/README.md)
2. 使用 `generate_perovskites.py` 生成自定义组成
3. 使用 `analyze_perovskites.py` 分析结果

### 第 3 天：进阶
1. 阅读 [PEROVSKITE_GENERATION_GUIDE.md](PEROVSKITE_GENERATION_GUIDE.md)
2. 尝试参数调优
3. 使用 MCTS 生成高质量结构

### 第 4 天：高级
1. 学习自定义评分函数
2. 集成能量预测模型（如 ALIGNN）
3. 针对特定应用优化

---

## 🤝 贡献和反馈

### 反馈渠道
- **GitHub Issues**：报告问题或建议
- **Pull Requests**：贡献代码或文档

### 欢迎贡献
- 新的钙钛矿组成
- 改进的评分函数
- 额外的分析功能
- 文档改进

---

## 📜 引用

如果在研究中使用了这些工具，请引用原始 CrystaLLM 论文：

```bibtex
@article{antunes2024crystal,
  title={Crystal structure generation with autoregressive large language modeling},
  author={Antunes, Luis M and Butler, Keith T and Grau-Crespo, Ricardo},
  journal={Nature Communications},
  volume={15},
  pages={10761},
  year={2024}
}
```

---

## 📞 获取帮助

### 文档
- [完整指南](PEROVSKITE_GENERATION_GUIDE.md) - 详细说明
- [中文 README](README_CN.md) - 快速入门
- [示例文档](examples/README.md) - 脚本使用

### 常见问题
查看 [PEROVSKITE_GENERATION_GUIDE.md](PEROVSKITE_GENERATION_GUIDE.md) 的"常见问题"章节

### 故障排除
查看 [examples/README.md](examples/README.md) 的"故障排除"章节

---

## 🌟 特别说明

本工具包是 CrystaLLM 项目的扩展，专门优化用于钙钛矿结构的生成和分析。

### 主要特性
- ✅ **完整的中文文档**（~1,800 行）
- ✅ **实用的自动化脚本**（~870 行）
- ✅ **一键运行演示**
- ✅ **专业的质量评估**
- ✅ **丰富的使用示例**

### 适用场景
- 🔬 材料研究和探索
- 📚 教学和学习
- 🏭 工业应用（太阳能电池、压电材料等）
- 💡 新材料设计

---

**开始使用：`cd examples && ./quickstart.sh`**

**祝您使用愉快！如果有帮助，请给项目一个 ⭐**
