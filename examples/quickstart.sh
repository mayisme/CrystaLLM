#!/bin/bash
# CrystaLLM 钙钛矿生成快速入门脚本
# 此脚本演示如何快速开始生成和分析钙钛矿结构

set -e  # 遇到错误时退出

echo "========================================"
echo "CrystaLLM 钙钛矿生成快速入门"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}项目目录: $PROJECT_ROOT${NC}"
echo ""

# 检查依赖
echo "步骤 1: 检查依赖..."
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}错误: 未找到 PyTorch${NC}"
    echo "请运行: pip install torch==2.0.1"
    exit 1
fi

if ! python -c "import pymatgen" 2>/dev/null; then
    echo -e "${YELLOW}警告: 未找到 pymatgen，正在安装...${NC}"
    pip install pymatgen
fi

if ! python -c "import pandas" 2>/dev/null; then
    echo -e "${YELLOW}警告: 未找到 pandas，正在安装...${NC}"
    pip install pandas
fi

if ! python -c "import tqdm" 2>/dev/null; then
    echo -e "${YELLOW}警告: 未找到 tqdm，正在安装...${NC}"
    pip install tqdm
fi

echo -e "${GREEN}✓ 依赖检查完成${NC}"
echo ""

# 检查模型
echo "步骤 2: 检查预训练模型..."
MODEL_DIR="crystallm_perov_5_small"

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/ckpt.pt" ]; then
    echo -e "${YELLOW}未找到钙钛矿模型，正在下载...${NC}"
    
    if [ ! -f "crystallm_perov_5_small.tar.gz" ]; then
        echo "下载模型文件..."
        python bin/download.py crystallm_perov_5_small.tar.gz
    fi
    
    echo "解压模型..."
    tar xzf crystallm_perov_5_small.tar.gz
    
    echo -e "${GREEN}✓ 模型下载完成${NC}"
else
    echo -e "${GREEN}✓ 找到模型: $MODEL_DIR${NC}"
fi
echo ""

# 创建输出目录
OUTPUT_DIR="quickstart_results"
mkdir -p "$OUTPUT_DIR"

echo "步骤 3: 生成钙钛矿结构..."
echo "组成: CaTiO3 (钛酸钙), SrTiO3 (钛酸锶), BaTiO3 (钛酸钡)"
echo "每种组成生成 5 个样本"
echo ""

# 使用生成脚本
python examples/generate_perovskites.py \
    --model "$MODEL_DIR" \
    --output "$OUTPUT_DIR/perovskites" \
    --compositions "CaTiO3:Pm-3m,SrTiO3:Pm-3m,BaTiO3:P4mm" \
    --num-samples 5 \
    --temperature 0.75 \
    --top-k 10 \
    --device cuda 2>/dev/null || \
python examples/generate_perovskites.py \
    --model "$MODEL_DIR" \
    --output "$OUTPUT_DIR/perovskites" \
    --compositions "CaTiO3:Pm-3m,SrTiO3:Pm-3m,BaTiO3:P4mm" \
    --num-samples 5 \
    --temperature 0.75 \
    --top-k 10 \
    --device cpu

echo ""
echo -e "${GREEN}✓ 结构生成完成${NC}"
echo ""

# 检查生成的文件
TOTAL_FILES=$(find "$OUTPUT_DIR/perovskites" -name "*.cif" -type f | wc -l)
echo "总共生成了 $TOTAL_FILES 个 CIF 文件"
echo ""

# 分析结构
echo "步骤 4: 分析生成的结构..."
echo ""

python examples/analyze_perovskites.py \
    --input "$OUTPUT_DIR/perovskites" \
    --output "$OUTPUT_DIR/analysis"

echo ""
echo -e "${GREEN}✓ 结构分析完成${NC}"
echo ""

# 显示结果摘要
echo "========================================"
echo "快速入门完成！"
echo "========================================"
echo ""
echo "结果位置:"
echo "  - 生成的结构: $OUTPUT_DIR/perovskites/"
echo "  - 分析报告: $OUTPUT_DIR/analysis/"
echo ""
echo "查看结果:"
echo "  1. 查看生成报告:"
echo "     cat $OUTPUT_DIR/perovskites/generation_report.txt"
echo ""
echo "  2. 查看分析报告:"
echo "     cat $OUTPUT_DIR/analysis/analysis_report.txt"
echo ""
echo "  3. 查看高质量结构列表:"
echo "     cat $OUTPUT_DIR/analysis/high_quality_structures.csv"
echo ""
echo "  4. 查看具体的 CIF 文件:"
echo "     ls $OUTPUT_DIR/perovskites/CaTiO3/processed/"
echo ""
echo "下一步:"
echo "  - 生成更多结构: python examples/generate_perovskites.py --help"
echo "  - 自定义分析: python examples/analyze_perovskites.py --help"
echo "  - 查看完整指南: cat PEROVSKITE_GENERATION_GUIDE.md"
echo ""
echo "========================================"
