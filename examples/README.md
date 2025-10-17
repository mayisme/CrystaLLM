# CrystaLLM 钙钛矿生成示例脚本

这个目录包含了使用 CrystaLLM 生成和分析稳定钙钛矿结构的示例脚本。

## 脚本概览

### 1. `generate_perovskites.py` - 批量生成钙钛矿结构

这个脚本可以批量生成多种钙钛矿组成的晶体结构。

**功能特性：**
- 支持自定义化学组成和空间群
- 预设常见钙钛矿组成列表
- 自动生成提示文件
- 批量调用 CrystaLLM 模型
- 自动后处理生成的结构
- 生成详细的生成报告

**基本用法：**
```bash
# 使用预设的常见钙钛矿列表
python generate_perovskites.py --model crystallm_perov_5_small --output perovskites/

# 列出所有预设的钙钛矿组成
python generate_perovskites.py --list-common

# 自定义组成和空间群
python generate_perovskites.py \
    --model crystallm_perov_5_small \
    --output my_perovskites/ \
    --compositions "CaTiO3:Pnma,SrTiO3:Pm-3m,BaTiO3:P4mm" \
    --num-samples 20

# 使用更保守的参数以提高稳定性
python generate_perovskites.py \
    --model crystallm_perov_5_small \
    --output stable_perovskites/ \
    --temperature 0.6 \
    --top-k 5 \
    --num-samples 50
```

**参数说明：**
- `--model, -m`: 训练好的模型目录路径（必需）
- `--output, -o`: 输出目录（默认：generated_perovskites）
- `--compositions, -c`: 自定义组成列表，格式："comp1:sg1,comp2:sg2"
- `--num-samples, -n`: 每种组成生成的样本数（默认：10）
- `--temperature, -t`: 采样温度，较低值更保守（默认：0.75）
- `--top-k, -k`: Top-K 采样参数（默认：10）
- `--device, -d`: 计算设备，cuda 或 cpu（默认：cuda）
- `--max-tokens`: 最大生成 token 数（默认：3000）
- `--list-common`: 列出常见钙钛矿组成并退出

**输出结构：**
```
output_directory/
├── composition1/
│   ├── raw/              # 原始生成的 CIF 文件
│   │   ├── sample_1.cif
│   │   ├── sample_2.cif
│   │   └── ...
│   └── processed/        # 后处理的 CIF 文件
│       ├── sample_1.cif
│       ├── sample_2.cif
│       └── ...
├── composition2/
│   └── ...
├── prompts/              # 生成的提示文件
│   ├── composition1_prompt.txt
│   └── ...
└── generation_report.txt # 生成任务报告
```

### 2. `analyze_perovskites.py` - 分析和筛选钙钛矿结构

这个脚本用于全面分析生成的钙钛矿结构，评估质量，并筛选最佳结构。

**功能特性：**
- 验证 CIF 文件有效性
- 计算键长合理性
- 评估空间群一致性
- 计算 Goldschmidt 容忍因子
- 综合质量评分
- 生成详细的分析报告
- 自动筛选高质量结构

**基本用法：**
```bash
# 分析目录中的所有结构
python analyze_perovskites.py --input perovskites/ --output analysis/

# 只分析有效结构
python analyze_perovskites.py \
    --input perovskites/ \
    --output analysis/ \
    --valid-only

# 设置质量阈值（0-100分）
python analyze_perovskites.py \
    --input perovskites/ \
    --output analysis/ \
    --min-quality 80

# 不递归搜索子目录
python analyze_perovskites.py \
    --input perovskites/CaTiO3/processed/ \
    --output analysis_catio3/ \
    --no-recursive
```

**参数说明：**
- `--input, -i`: 包含 CIF 文件的输入目录（必需）
- `--output, -o`: 输出目录（默认：analysis_results）
- `--no-recursive`: 不递归搜索子目录
- `--valid-only`: 只显示有效结构
- `--min-quality`: 最小质量分数阈值（0-100）

**分析指标：**

1. **结构有效性（40分）**
   - CIF 文件格式正确
   - 可以构建有效的晶体结构
   - 原子坐标合理

2. **空间群一致性（20分）**
   - 生成的结构符合声明的空间群
   - 对称操作正确

3. **原子位点重数一致性（15分）**
   - 原子占据合理
   - 重数与对称性匹配

4. **键长合理性（15分）**
   - 原子间距在合理范围内
   - 无异常的近距离接触

5. **容忍因子稳定性（10分）**
   - Goldschmidt 容忍因子在 0.8-1.0 范围内
   - 适用于 ABX₃ 型钙钛矿

**输出文件：**
```
output_directory/
├── analysis_full.csv              # 所有结构的完整分析数据
├── high_quality_structures.csv    # 高质量结构列表（≥80分）
└── analysis_report.txt            # 详细的文本报告
```

**分析报告包含：**
- 总体统计
- 有效性统计
- 质量分数分布
- 容忍因子统计
- 晶系分布
- 前10个最高质量结构

## 完整工作流程示例

下面是一个从生成到分析的完整工作流程：

### 第1步：下载预训练模型

```bash
cd /path/to/CrystaLLM

# 下载钙钛矿专用模型
python bin/download.py crystallm_perov_5_small.tar.gz

# 解压模型
tar xvf crystallm_perov_5_small.tar.gz
```

### 第2步：批量生成钙钛矿结构

```bash
cd examples

# 生成常见钙钛矿（自动使用预设列表）
python generate_perovskites.py \
    --model ../crystallm_perov_5_small \
    --output ../results/perovskites \
    --num-samples 20 \
    --temperature 0.7 \
    --top-k 8
```

这将生成包括以下组成的结构：
- CaTiO₃ (钛酸钙)
- SrTiO₃ (钛酸锶)
- BaTiO₃ (钛酸钡)
- PbTiO₃ (钛酸铅)
- LaAlO₃ (铝酸镧)
- CsPbI₃ (碘化铯铅)
- 等等...

### 第3步：分析生成的结构

```bash
# 全面分析所有生成的结构
python analyze_perovskites.py \
    --input ../results/perovskites \
    --output ../results/analysis
```

### 第4步：查看结果

```bash
# 查看分析报告
cat ../results/analysis/analysis_report.txt

# 查看高质量结构列表
cat ../results/analysis/high_quality_structures.csv

# 或使用 pandas 进行进一步分析
python << EOF
import pandas as pd

# 读取完整分析数据
df = pd.read_csv('../results/analysis/analysis_full.csv')

# 筛选优秀结构
excellent = df[df['quality_score'] >= 80]

# 按组成分组统计
by_formula = df.groupby('formula')['quality_score'].agg(['mean', 'max', 'count'])
print(by_formula)

# 找到最稳定的钙钛矿（基于容忍因子）
stable = df[(df['tolerance_factor'] >= 0.8) & (df['tolerance_factor'] <= 1.0)]
stable = stable.sort_values('quality_score', ascending=False)
print(stable[['formula', 'space_group', 'tolerance_factor', 'quality_score']].head(10))
EOF
```

### 第5步：提取最佳结构

```bash
# 根据分析结果，复制高质量结构到新目录
mkdir -p ../results/best_structures

# 使用 Python 脚本提取
python << EOF
import pandas as pd
import shutil
from pathlib import Path

df = pd.read_csv('../results/analysis/high_quality_structures.csv')

output_dir = Path('../results/best_structures')
output_dir.mkdir(exist_ok=True)

for idx, row in df.iterrows():
    src = Path(row['path'])
    dst = output_dir / f"{row['formula']}_{row['space_group']}_{idx}.cif"
    shutil.copy(src, dst)
    print(f"Copied: {dst.name}")

print(f"\nTotal: {len(df)} high-quality structures")
EOF
```

## 高级用法

### 1. 针对特定应用生成钙钛矿

**太阳能电池材料（需要合适的带隙）：**
```bash
python generate_perovskites.py \
    --model ../crystallm_perov_5_small \
    --output ../results/solar_cell_perovskites \
    --compositions "CsPbI3:Pm-3m,CsPbBr3:Pm-3m,CsSnI3:Pnma" \
    --num-samples 50 \
    --temperature 0.65
```

**压电材料（需要非中心对称空间群）：**
```bash
python generate_perovskites.py \
    --model ../crystallm_perov_5_small \
    --output ../results/piezo_perovskites \
    --compositions "BaTiO3:P4mm,PbTiO3:P4mm,KNbO3:Amm2" \
    --num-samples 30 \
    --temperature 0.7
```

### 2. 使用自定义评分标准筛选

```python
# custom_filter.py
import pandas as pd

# 读取分析结果
df = pd.read_csv('analysis/analysis_full.csv')

# 自定义筛选条件
filtered = df[
    (df['quality_score'] >= 80) &               # 高质量
    (df['tolerance_factor'] >= 0.85) &          # 稳定的容忍因子
    (df['tolerance_factor'] <= 0.95) &
    (df['crystal_system'] == 'cubic') &         # 立方相
    (df['bond_length_score'] >= 0.98)           # 优秀的键长
]

print(f"Found {len(filtered)} structures matching criteria")
filtered.to_csv('custom_filtered.csv', index=False)
```

### 3. 批量转换为其他格式

```python
# convert_structures.py
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar
from pathlib import Path

input_dir = Path('best_structures')
output_dir = Path('poscar_files')
output_dir.mkdir(exist_ok=True)

for cif_file in input_dir.glob('*.cif'):
    parser = CifParser(str(cif_file))
    structure = parser.get_structures()[0]
    
    # 转换为 VASP POSCAR 格式
    poscar = Poscar(structure)
    output_file = output_dir / f"{cif_file.stem}.POSCAR"
    poscar.write_file(str(output_file))
    
    print(f"Converted: {cif_file.name} -> {output_file.name}")
```

## 注意事项

1. **计算资源**：生成大量结构需要足够的 GPU 内存。如果遇到内存问题，可以：
   - 减少 `--num-samples`
   - 使用 `--device cpu`（速度会慢很多）
   - 分批次生成

2. **模型选择**：
   - `crystallm_perov_5_small`: 快速，适合探索
   - `crystallm_perov_5_large`: 更高质量，但需要更多资源
   - `crystallm_v1_small/large`: 通用模型，可生成各种晶体

3. **参数调优**：
   - **追求稳定性**: 降低 `temperature` (0.6-0.7), 减小 `top_k` (5-8)
   - **追求多样性**: 提高 `temperature` (0.9-1.1), 增大 `top_k` (15-30)
   - **平衡**: 默认参数 (`temperature=0.75`, `top_k=10`)

4. **验证结果**：
   - 始终检查生成结构的有效性
   - 使用 DFT 计算验证重要的结构
   - 容忍因子只是初步估计，不能完全保证稳定性

## 故障排除

**问题：生成的结构大多无效**
- 解决：降低 `temperature`，减小 `top_k`，增加 `num_samples` 后筛选

**问题：生成速度太慢**
- 解决：确保使用 GPU (`--device cuda`)，或使用更小的模型

**问题：内存不足错误**
- 解决：减少 `batch_size`（需要修改模型配置），或使用 CPU

**问题：找不到模型文件**
- 解决：确保模型路径正确，模型目录中应包含 `ckpt.pt` 文件

**问题：分析脚本报错**
- 解决：确保安装了所有依赖：`pip install pymatgen pandas tqdm`

## 扩展阅读

- **CrystaLLM 主文档**: [../README.md](../README.md)
- **详细指南**: [../PEROVSKITE_GENERATION_GUIDE.md](../PEROVSKITE_GENERATION_GUIDE.md)
- **论文**: [Crystal Structure Generation with Autoregressive Large Language Modeling](https://www.nature.com/articles/s41467-024-54639-7)

## 贡献

欢迎贡献新的示例脚本或改进现有脚本！

## 许可证

这些示例脚本遵循与 CrystaLLM 项目相同的许可证。
