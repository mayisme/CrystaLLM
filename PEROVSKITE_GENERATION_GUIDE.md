# CrystaLLM：生成稳定钙钛矿结构的完整指南

## 目录
1. [项目概述](#项目概述)
2. [核心原理](#核心原理)
3. [代码架构分析](#代码架构分析)
4. [生成稳定钙钛矿结构的步骤](#生成稳定钙钛矿结构的步骤)
5. [关键参数调优](#关键参数调优)
6. [实践案例](#实践案例)
7. [高级技术：MCTS解码](#高级技术mcts解码)

---

## 项目概述

CrystaLLM 是一个基于 GPT-2 架构的大型语言模型，专门用于生成晶体结构的 CIF (Crystallographic Information File) 格式文件。该项目发表于 Nature Communications，可以根据给定的化学组成和空间群来生成晶体结构。

### 主要特性
- **基于 Transformer 架构**：采用 GPT-2 模型进行序列生成
- **CIF 格式输出**：直接生成标准的晶体学信息文件
- **条件生成**：支持指定化学组成和空间群
- **质量评估**：内置多种结构验证和评分机制
- **MCTS 优化**：支持蒙特卡洛树搜索以生成更稳定的结构

---

## 核心原理

### 1. 模型架构

CrystaLLM 将 CIF 文件视为文本序列，使用自回归语言模型进行生成：

```
输入提示 (Prompt) → GPT-2 模型 → 生成的 CIF 文件
```

**关键组件：**
- **Tokenizer** (`crystallm/_tokenizer.py`)：将 CIF 文件转换为 token 序列
- **GPT 模型** (`crystallm/_model.py`)：基于 GPT-2 的生成模型
- **配置系统** (`crystallm/_configuration.py`)：统一的参数管理

### 2. CIF 文件格式

生成的 CIF 文件包含以下关键信息：
- **data_** 行：化学组成（如 `data_CaTiO3`）
- **原子属性**：电负性、原子半径、离子半径
- **空间群**：晶体对称性信息
- **晶格参数**：a, b, c, α, β, γ
- **原子坐标**：分数坐标和占据度

### 3. 钙钛矿结构特点

钙钛矿（Perovskite）通常具有 ABX₃ 结构：
- **A 位**：较大的阳离子（如 Ca²⁺, Pb²⁺）
- **B 位**：较小的阳离子（如 Ti⁴⁺, Sn⁴⁺）
- **X 位**：阴离子（如 O²⁻, I⁻）

常见空间群：Pm-3m (立方), P4mm (四方), Pnma (正交) 等

---

## 代码架构分析

### 核心模块

#### 1. `bin/sample.py` - 随机采样生成
**用途**：使用训练好的模型进行随机采样生成 CIF 文件

**关键参数：**
```python
out_dir: str        # 训练模型所在目录
start: str          # 提示文本（可以是文件路径）
num_samples: int    # 生成样本数量
max_new_tokens: int # 每个样本最大 token 数
temperature: float  # 采样温度（控制随机性）
top_k: int         # Top-K 采样参数
```

**工作流程：**
1. 加载训练好的模型检查点
2. 编码提示文本
3. 生成 CIF 序列
4. 解码输出

#### 2. `bin/make_prompt_file.py` - 生成提示文件
**用途**：根据化学组成和可选的空间群创建提示

**核心函数：**
```python
def get_prompt(comp, sg=None):
    comp_str = comp.formula.replace(" ", "")
    if sg is not None:
        # 包含空间群的提示
        block = get_atomic_props_block_for_formula(comp_str)
        return f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
    else:
        # 仅组成的提示
        return f"data_{comp_str}\n"
```

**示例：**
- 不指定空间群：`data_CaTiO3\n`
- 指定空间群：包含完整的原子属性块和空间群信息

#### 3. `bin/mcts.py` - 蒙特卡洛树搜索
**用途**：使用 MCTS 算法优化生成过程，提高结构质量

**关键参数：**
```python
tree_width: int           # 树的宽度（每层展开的节点数）
max_depth: int            # 最大搜索深度
num_simulations: int      # 模拟次数
c: float                  # 探索/利用平衡参数
bond_length_acceptability_cutoff: float  # 键长可接受性阈值
reward_k: float           # 奖励函数常数
scorer: str               # 评分器类型（'zmq' 或 'random'）
```

**MCTS 评分系统：**
- **ZMQScorer**：通过 ZMQ 协议连接外部评分器（如 ALIGNN 能量预测模型）
- **RandomScorer**：随机评分（用于测试）

#### 4. `bin/evaluate_cifs.py` - 结构评估
**用途**：评估生成的 CIF 文件的质量

**评估指标：**
- **空间群一致性**：生成的结构是否符合声明的空间群
- **原子位点重数一致性**：原子占据是否合理
- **键长合理性**：原子间距是否在合理范围内
- **结构有效性**：是否可以构建有效的晶体结构

#### 5. `crystallm/_utils.py` - 工具函数
**核心功能：**
- `get_atomic_props_block_for_formula()`: 生成原子属性块
- `extract_space_group_symbol()`: 提取空间群符号
- `is_valid()`: 验证 CIF 结构有效性
- `bond_length_reasonableness_score()`: 计算键长合理性分数

---

## 生成稳定钙钛矿结构的步骤

### 方法一：使用预训练模型快速生成

#### 步骤 1：下载预训练模型
```bash
# 下载钙钛矿专用小模型
python bin/download.py crystallm_perov_5_small.tar.gz

# 解压模型
tar xvf crystallm_perov_5_small.tar.gz
```

#### 步骤 2：创建钙钛矿组成的提示文件
```bash
# 示例：生成 CaTiO3（钛酸钙）钙钛矿
python bin/make_prompt_file.py CaTiO3 prompts/catio3_prompt.txt

# 指定空间群（Pm-3m 是立方钙钛矿的典型空间群）
python bin/make_prompt_file.py CaTiO3 prompts/catio3_cubic_prompt.txt --spacegroup Pm-3m

# 其他钙钛矿例子
python bin/make_prompt_file.py SrTiO3 prompts/srtio3_prompt.txt --spacegroup Pm-3m
python bin/make_prompt_file.py BaTiO3 prompts/batio3_prompt.txt --spacegroup P4mm
python bin/make_prompt_file.py MAPbI3 prompts/mapbi3_prompt.txt  # 有机-无机杂化钙钛矿
```

**重要提示**：
- 元素必须按电负性排序（模型训练时的顺序）
- 使用单元胞组成（不是约化式）

#### 步骤 3：使用模型生成 CIF 文件
```bash
# 基本生成（生成到控制台）
python bin/sample.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:prompts/catio3_cubic_prompt.txt \
    num_samples=10 \
    max_new_tokens=3000 \
    temperature=0.8 \
    top_k=10 \
    device=cuda

# 保存到文件
python bin/sample.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:prompts/catio3_cubic_prompt.txt \
    num_samples=10 \
    max_new_tokens=3000 \
    temperature=0.8 \
    top_k=10 \
    device=cuda \
    target=file
```

**参数说明：**
- `temperature`: 较低值（0.6-0.8）生成更保守/稳定的结构
- `top_k`: 较小值（5-15）提高生成质量
- `num_samples`: 生成多个样本以选择最佳结构

#### 步骤 4：后处理生成的 CIF 文件
```bash
# 假设生成的原始文件在 raw_cifs/ 目录
python bin/postprocess.py raw_cifs/ processed_cifs/
```

后处理会：
- 规范化 CIF 格式
- 修正对称操作
- 标准化数值精度

#### 步骤 5：评估生成的结构
```bash
# 将处理后的 CIF 文件打包
tar czf processed_perovskites.tar.gz processed_cifs/

# 评估结构质量
python bin/evaluate_cifs.py \
    processed_perovskites.tar.gz \
    --out perovskite_evaluation.csv \
    --workers 4
```

**评估输出示例：**
```
space group consistent: 95/100 (0.950)
atom site multiplicity consistent: 98/100 (0.980)
avg. bond length reasonableness score: 0.9845 ± 0.0412
bond lengths reasonable: 94/100 (0.940)
num valid: 92/100 (0.92)
```

---

### 方法二：使用 MCTS 生成高质量结构

MCTS（蒙特卡洛树搜索）通过搜索算法优化生成过程，可以显著提高结构质量。

#### 步骤 1：设置外部评分器（可选但推荐）

**使用 ALIGNN 作为能量预测评分器：**

创建 `alignn_scorer.py`：
```python
import zmq
import json
from alignn.pretrained import get_prediction
from pymatgen.io.cif import CifParser
from io import StringIO

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("ALIGNN scorer listening on port 5555...")

while True:
    message = socket.recv_string()
    cif_content = json.loads(message)["cif"]
    
    try:
        # 使用 ALIGNN 预测形成能
        parser = CifParser(StringIO(cif_content))
        structure = parser.get_structures()[0]
        
        # 预测形成能（越低越稳定）
        formation_energy = get_prediction(
            structure, 
            model_name="formation_energy_peratom"
        )
        
        # 转换为奖励（负的形成能）
        reward = -formation_energy
        
        response = {"score": reward}
    except Exception as e:
        response = {"score": 0.0, "error": str(e)}
    
    socket.send_string(json.dumps(response))
```

在另一个终端启动评分器：
```bash
python alignn_scorer.py
```

#### 步骤 2：使用 MCTS 生成
```bash
python bin/mcts.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:prompts/catio3_cubic_prompt.txt \
    tree_width=10 \
    max_depth=1000 \
    num_simulations=200 \
    c=5.0 \
    bond_length_acceptability_cutoff=1.0 \
    reward_k=2.0 \
    mcts_out_dir=mcts_perovskites \
    scorer=zmq \
    scorer_host=localhost \
    scorer_port=5555 \
    selector=puct \
    device=cuda
```

**MCTS 参数详解：**
- `tree_width=10`: 每个节点展开 10 个子节点
- `num_simulations=200`: 执行 200 次模拟
- `c=5.0`: PUCT 探索常数（越大越倾向探索）
- `bond_length_acceptability_cutoff=1.0`: 键长合理性阈值
- `reward_k=2.0`: 奖励缩放因子
- `selector=puct`: 使用 PUCT 选择策略（其他选项：uct, greedy）

#### 步骤 3：分析 MCTS 结果
```bash
# MCTS 会在 mcts_out_dir 中保存生成的 CIF 文件
# 选择得分最高的结构
ls -lh mcts_perovskites/
```

---

### 方法三：训练专用钙钛矿模型

如果您有特定的钙钛矿数据集，可以训练专门的模型：

#### 步骤 1：准备钙钛矿 CIF 数据
```bash
# 假设您的钙钛矿 CIF 文件在 perovskite_cifs/ 目录
python bin/prepare_custom.py perovskite_cifs/ perovskite_cifs.tar.gz

# 转换为 pickle 格式
python bin/tar_to_pickle.py perovskite_cifs.tar.gz perovskite_cifs.pkl.gz
```

#### 步骤 2：预处理和分割数据
```bash
# 预处理
python bin/preprocess.py \
    perovskite_cifs.pkl.gz \
    --out perovskite_prep.pkl.gz \
    --workers 4

# 分割数据集
python bin/split.py perovskite_prep.pkl.gz \
    --train_out perovskite_train.pkl.gz \
    --val_out perovskite_val.pkl.gz \
    --test_out perovskite_test.pkl.gz
```

#### 步骤 3：标记化
```bash
python bin/tokenize_cifs.py \
    --train_fname perovskite_train.pkl.gz \
    --val_fname perovskite_val.pkl.gz \
    --out_dir tokens_perovskite/ \
    --workers 4
```

#### 步骤 4：训练模型
创建配置文件 `config/my_perovskite.yaml`：
```yaml
out_dir: 'out/my_perovskite_model'
dataset: 'tokens_perovskite/'
validate: True

# 模型架构
n_layer: 12
n_head: 12
n_embd: 768
dropout: 0.1
block_size: 2048

# 训练参数
batch_size: 32
gradient_accumulation_steps: 4
learning_rate: 1e-3
max_iters: 5000
warmup_iters: 200

# 学习率衰减
decay_lr: True
lr_decay_iters: 5000
min_lr: 1e-4

# 评估
eval_interval: 250
always_save_checkpoint: True
```

训练：
```bash
python bin/train.py --config=config/my_perovskite.yaml
```

---

## 关键参数调优

### 1. 采样参数

| 参数 | 推荐值（稳定性） | 推荐值（多样性） | 作用 |
|------|-----------------|-----------------|------|
| `temperature` | 0.6-0.8 | 1.0-1.2 | 控制随机性 |
| `top_k` | 5-10 | 20-50 | 候选 token 数量 |
| `num_samples` | 50-100 | 10-20 | 生成样本数 |
| `max_new_tokens` | 3000-5000 | 2000-3000 | 最大生成长度 |

### 2. MCTS 参数

**追求稳定性：**
- `num_simulations`: 300-500（更多模拟）
- `tree_width`: 5-10（较窄的搜索）
- `c`: 3.0-5.0（平衡探索）

**追求多样性：**
- `num_simulations`: 100-200
- `tree_width`: 15-20（更宽的搜索）
- `c`: 5.0-10.0（更多探索）

### 3. 结构筛选策略

生成多个结构后，按以下标准筛选：

1. **有效性检查**：`is_valid == True`
2. **空间群一致性**：`space_group_consistent == True`
3. **键长合理性**：`bond_length_reasonableness_score > 0.95`
4. **能量评分**（如使用 ALIGNN）：选择形成能最低的结构
5. **结构对称性**：检查是否符合预期的钙钛矿对称性

---

## 实践案例

### 案例 1：生成立方相 CaTiO3

```bash
# 1. 创建提示（Pm-3m 空间群）
python bin/make_prompt_file.py CaTiO3 catio3_cubic.txt --spacegroup "Pm-3m"

# 2. 生成 50 个候选结构
python bin/sample.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:catio3_cubic.txt \
    num_samples=50 \
    temperature=0.7 \
    top_k=8 \
    target=file \
    device=cuda

# 3. 后处理
mkdir raw_catio3
mv sample_*.cif raw_catio3/
python bin/postprocess.py raw_catio3/ processed_catio3/

# 4. 评估
tar czf catio3_gen.tar.gz processed_catio3/
python bin/evaluate_cifs.py catio3_gen.tar.gz -o catio3_eval.csv

# 5. 选择最佳结构
python -c "
import pandas as pd
df = pd.read_csv('catio3_eval.csv')
valid = df[df['is_valid'] == True]
best = valid.iloc[0]
print(f'Best structure: {best[\"comp\"]}, SG: {best[\"sg\"]}')
"
```

### 案例 2：使用 MCTS 生成高稳定性钙钛矿太阳能电池材料

```bash
# 目标：MAPbI3（甲胺铅碘钙钛矿）
python bin/make_prompt_file.py CH3NH3PbI3 mapbi3.txt --spacegroup "P4mm"

# 启动 ALIGNN 评分器（另一终端）
python alignn_scorer.py

# 使用 MCTS 生成
python bin/mcts.py \
    out_dir=crystallm_perov_5_small \
    start=FILE:mapbi3.txt \
    tree_width=8 \
    num_simulations=300 \
    c=4.0 \
    scorer=zmq \
    mcts_out_dir=mapbi3_mcts \
    device=cuda

# 生成的最佳结构会保存在 mapbi3_mcts/ 目录
```

### 案例 3：批量生成多种钙钛矿组成

```bash
# 创建组成列表文件 perovskite_compositions.txt
# CaTiO3
# SrTiO3
# BaTiO3
# PbTiO3
# LaAlO3

# 批量生成提示
while read comp; do
    python bin/make_prompt_file.py "$comp" "prompts/${comp}_prompt.txt" --spacegroup "Pm-3m"
done < perovskite_compositions.txt

# 批量生成结构
for prompt in prompts/*_prompt.txt; do
    base=$(basename "$prompt" _prompt.txt)
    python bin/sample.py \
        out_dir=crystallm_perov_5_small \
        start="FILE:$prompt" \
        num_samples=20 \
        temperature=0.75 \
        target=file \
        device=cuda
    mkdir -p "generated/$base"
    mv sample_*.cif "generated/$base/"
done
```

---

## 高级技术：MCTS解码

### MCTS 算法原理

CrystaLLM 中的 MCTS 实现（`crystallm/_mcts.py`）包含以下核心组件：

1. **TreeBuilder**：构建搜索树
   - `ContextSensitiveTreeBuilder`：根据上下文智能构建节点

2. **NodeSelector**：选择展开哪个节点
   - `PUCTSelector`：PUCT (Predictor + UCT) 策略
   - `UCTSelector`：经典 UCT 策略
   - `GreedySelector`：贪心选择

3. **Evaluator**：评估节点质量
   - 使用外部评分器（如 ALIGNN）
   - 考虑键长合理性
   - 计算奖励值

### 自定义评分函数

您可以实现自己的评分器来优化特定属性：

```python
# custom_scorer.py
import zmq
import json
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
from io import StringIO

class StabilityScorer:
    """基于多个稳定性指标的综合评分器"""
    
    def __init__(self):
        self.matcher = StructureMatcher()
    
    def calculate_score(self, cif_content):
        try:
            parser = CifParser(StringIO(cif_content))
            structure = parser.get_structures()[0]
            
            # 1. 形成能（使用 ALIGNN 或其他 ML 模型）
            formation_energy = self.predict_formation_energy(structure)
            
            # 2. 带隙（用于太阳能电池材料）
            band_gap = self.predict_band_gap(structure)
            
            # 3. 结构对称性评分
            symmetry_score = self.evaluate_symmetry(structure)
            
            # 4. 化学稳定性（Goldschmidt 容忍因子等）
            tolerance_factor = self.calculate_tolerance_factor(structure)
            
            # 综合评分
            score = (
                -formation_energy * 0.4 +    # 低形成能更好
                band_gap * 0.3 +               # 合适的带隙
                symmetry_score * 0.2 +         # 高对称性
                tolerance_factor * 0.1         # 合理的容忍因子
            )
            
            return score
            
        except Exception as e:
            return 0.0
    
    def predict_formation_energy(self, structure):
        # 实现或调用预训练模型
        pass
    
    def predict_band_gap(self, structure):
        # 实现或调用预训练模型
        pass
    
    def evaluate_symmetry(self, structure):
        # 评估结构对称性
        pass
    
    def calculate_tolerance_factor(self, structure):
        # 计算 Goldschmidt 容忍因子
        # t = (r_A + r_X) / (√2 * (r_B + r_X))
        pass

# ZMQ 服务器
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

scorer = StabilityScorer()

while True:
    message = socket.recv_string()
    cif_content = json.loads(message)["cif"]
    score = scorer.calculate_score(cif_content)
    socket.send_string(json.dumps({"score": score}))
```

### MCTS 可视化和调试

```python
# mcts_analysis.py
import os
import pandas as pd
from crystallm import CIFTokenizer
from pymatgen.io.cif import CifParser

def analyze_mcts_output(mcts_dir):
    """分析 MCTS 输出目录中的所有结构"""
    
    results = []
    tokenizer = CIFTokenizer()
    
    for fname in os.listdir(mcts_dir):
        if not fname.endswith('.cif'):
            continue
        
        path = os.path.join(mcts_dir, fname)
        with open(path, 'r') as f:
            cif_content = f.read()
        
        try:
            parser = CifParser.from_string(cif_content)
            structure = parser.get_structures()[0]
            
            results.append({
                'file': fname,
                'formula': structure.composition.reduced_formula,
                'space_group': structure.get_space_group_info()[0],
                'volume': structure.volume,
                'density': structure.density,
                'n_sites': len(structure),
            })
        except:
            pass
    
    df = pd.DataFrame(results)
    print(df)
    return df

# 使用
df = analyze_mcts_output('mcts_perovskites/')
df.to_csv('mcts_analysis.csv', index=False)
```

---

## 总结

使用 CrystaLLM 生成稳定钙钛矿结构的关键要点：

1. **选择合适的模型**：使用 `crystallm_perov_5_small/large` 预训练模型
2. **精确的提示**：包含化学组成和目标空间群
3. **参数调优**：较低的 temperature 和 top_k 提高稳定性
4. **批量生成**：生成多个候选结构以选择最佳
5. **质量评估**：使用内置评估工具筛选有效结构
6. **高级优化**：使用 MCTS + 外部评分器生成高质量结构

## 常见问题

**Q: 如何判断生成的钙钛矿结构是否稳定？**
A: 检查以下指标：
- 形成能（使用 DFT 或 ML 模型预测）
- Goldschmidt 容忍因子（0.8-1.0 为稳定）
- 键长合理性分数 > 0.95
- 空间群一致性

**Q: 生成的结构与已知钙钛矿差异很大怎么办？**
A: 
- 降低 temperature 到 0.6-0.7
- 在提示中明确指定空间群
- 使用 MCTS 并配置合适的评分器
- 增加生成样本数量

**Q: 如何生成特定应用的钙钛矿（如太阳能电池）？**
A: 
- 使用领域特定的评分函数（考虑带隙、稳定性等）
- 准备该领域的训练数据重新训练模型
- 使用 MCTS 并优化目标属性

**Q: 可以生成有机-无机杂化钙钛矿吗？**
A: 可以，但需要注意：
- 确保训练数据包含有机组分
- 有机分子的命名要一致
- 可能需要更长的 max_new_tokens

---

## 参考资源

- **论文**：[Crystal Structure Generation with Autoregressive Large Language Modeling](https://www.nature.com/articles/s41467-024-54639-7)
- **原始代码**：[CrystaLLM GitHub](https://github.com/lantunes/CrystaLLM)
- **相关工具**：
  - [ALIGNN](https://github.com/usnistgov/alignn) - 图神经网络材料属性预测
  - [Pymatgen](https://pymatgen.org/) - 材料分析 Python 库
  - [Materials Project](https://materialsproject.org/) - 材料数据库

---

*本指南由 AI 助手生成，用于帮助研究人员使用 CrystaLLM 生成稳定的钙钛矿结构。*
