#!/usr/bin/env python3
"""
示例脚本：分析和筛选钙钛矿结构

此脚本用于分析生成的钙钛矿结构，评估其质量，并根据多种标准筛选最佳结构。

功能：
- 验证 CIF 文件有效性
- 计算键长合理性
- 评估空间群一致性
- 计算 Goldschmidt 容忍因子
- 预测形成能（如果有模型）
- 生成质量报告和可视化

用法：
    python analyze_perovskites.py --input perovskites/ --output analysis/
"""

import argparse
import os
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    print("错误: 需要安装 pymatgen")
    print("运行: pip install pymatgen")
    sys.exit(1)

from crystallm import (
    is_valid,
    is_sensible,
    is_space_group_consistent,
    is_atom_site_multiplicity_consistent,
    bond_length_reasonableness_score,
    extract_space_group_symbol,
    extract_volume,
)


def calculate_goldschmidt_tolerance_factor(structure):
    """
    计算钙钛矿的 Goldschmidt 容忍因子
    
    对于 ABX3 钙钛矿:
    t = (r_A + r_X) / (√2 * (r_B + r_X))
    
    稳定的钙钛矿通常 0.8 < t < 1.0
    
    Args:
        structure: Pymatgen Structure 对象
    
    Returns:
        容忍因子值，如果无法计算则返回 None
    """
    try:
        # 获取组成元素
        comp = structure.composition
        elements = list(comp.elements)
        
        # 简化假设：按含量排序，最多的是 X，中间是 A，最少的是 B
        element_amounts = [(str(el), comp.get_atomic_fraction(el)) for el in elements]
        element_amounts.sort(key=lambda x: x[1], reverse=True)
        
        if len(element_amounts) < 3:
            return None
        
        # 对于 ABX3，X 的含量应该最多
        X_symbol = element_amounts[0][0]
        A_symbol = element_amounts[1][0]
        B_symbol = element_amounts[2][0]
        
        # 获取离子半径（假设常见氧化态）
        # 这是简化版本，实际应该根据具体氧化态
        ionic_radii = {
            'O': 1.40,  # O2-
            'F': 1.33,  # F-
            'Cl': 1.81, # Cl-
            'Br': 1.96, # Br-
            'I': 2.20,  # I-
            'Ca': 1.00, # Ca2+
            'Sr': 1.18, # Sr2+
            'Ba': 1.35, # Ba2+
            'Pb': 1.19, # Pb2+
            'Cs': 1.67, # Cs+
            'Ti': 0.605, # Ti4+
            'Sn': 0.69,  # Sn4+
            'Ge': 0.53,  # Ge4+
            'Zr': 0.72,  # Zr4+
        }
        
        r_A = ionic_radii.get(A_symbol)
        r_B = ionic_radii.get(B_symbol)
        r_X = ionic_radii.get(X_symbol)
        
        if r_A is None or r_B is None or r_X is None:
            return None
        
        # 计算容忍因子
        t = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
        
        return t
        
    except Exception as e:
        return None


def analyze_structure(cif_path):
    """
    全面分析单个 CIF 结构
    
    Args:
        cif_path: CIF 文件路径
    
    Returns:
        包含分析结果的字典
    """
    results = {
        'file': os.path.basename(cif_path),
        'path': str(cif_path),
    }
    
    try:
        # 读取 CIF 文件
        with open(cif_path, 'r') as f:
            cif_content = f.read()
        
        # 基本验证
        results['is_sensible'] = is_sensible(cif_content)
        results['is_valid'] = is_valid(cif_content, bond_length_acceptability_cutoff=1.0)
        results['sg_consistent'] = is_space_group_consistent(cif_content)
        results['multiplicity_consistent'] = is_atom_site_multiplicity_consistent(cif_content)
        results['bond_length_score'] = bond_length_reasonableness_score(cif_content)
        
        # 提取基本信息
        try:
            results['space_group'] = extract_space_group_symbol(cif_content)
        except:
            results['space_group'] = None
        
        try:
            results['volume'] = extract_volume(cif_content)
        except:
            results['volume'] = None
        
        # 使用 Pymatgen 分析
        parser = CifParser.from_string(cif_content)
        structure = parser.get_structures()[0]
        
        results['formula'] = structure.composition.reduced_formula
        results['formula_full'] = structure.composition.formula
        results['density'] = structure.density
        results['n_sites'] = len(structure)
        results['n_elements'] = len(structure.composition.elements)
        
        # 对称性分析
        try:
            sga = SpacegroupAnalyzer(structure)
            results['sg_number'] = sga.get_space_group_number()
            results['sg_symbol'] = sga.get_space_group_symbol()
            results['crystal_system'] = sga.get_crystal_system()
            results['point_group'] = sga.get_point_group_symbol()
        except:
            pass
        
        # 钙钛矿特定分析
        if results.get('n_elements') == 3:  # ABX3 型
            tolerance = calculate_goldschmidt_tolerance_factor(structure)
            results['tolerance_factor'] = tolerance
            
            if tolerance is not None:
                # 容忍因子稳定性评估
                if 0.8 <= tolerance <= 1.0:
                    results['tolerance_stability'] = 'stable'
                elif 0.7 <= tolerance < 0.8:
                    results['tolerance_stability'] = 'marginally_stable'
                else:
                    results['tolerance_stability'] = 'unstable'
        
        # 晶格参数
        lattice = structure.lattice
        results['a'] = lattice.a
        results['b'] = lattice.b
        results['c'] = lattice.c
        results['alpha'] = lattice.alpha
        results['beta'] = lattice.beta
        results['gamma'] = lattice.gamma
        
        # 综合质量分数（0-100）
        quality_score = 0
        if results['is_valid']:
            quality_score += 40
        if results['sg_consistent']:
            quality_score += 20
        if results['multiplicity_consistent']:
            quality_score += 15
        if results.get('bond_length_score', 0) >= 0.95:
            quality_score += 15
        if results.get('tolerance_stability') == 'stable':
            quality_score += 10
        
        results['quality_score'] = quality_score
        results['status'] = 'success'
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        results['quality_score'] = 0
    
    return results


def analyze_directory(input_dir, recursive=True):
    """
    分析目录中的所有 CIF 文件
    
    Args:
        input_dir: 输入目录路径
        recursive: 是否递归搜索子目录
    
    Returns:
        包含所有分析结果的 DataFrame
    """
    input_path = Path(input_dir)
    
    # 查找所有 CIF 文件
    if recursive:
        cif_files = list(input_path.rglob('*.cif'))
    else:
        cif_files = list(input_path.glob('*.cif'))
    
    print(f"找到 {len(cif_files)} 个 CIF 文件")
    
    if len(cif_files) == 0:
        print("警告: 未找到 CIF 文件")
        return pd.DataFrame()
    
    # 分析每个文件
    results = []
    for cif_file in tqdm(cif_files, desc="分析结构"):
        result = analyze_structure(cif_file)
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def generate_report(df, output_dir):
    """
    生成分析报告
    
    Args:
        df: 分析结果 DataFrame
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存完整数据
    csv_file = output_path / 'analysis_full.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n完整分析结果已保存到: {csv_file}")
    
    # 生成文本报告
    report_file = output_path / 'analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("钙钛矿结构分析报告\n")
        f.write("="*70 + "\n\n")
        
        # 总体统计
        f.write("总体统计:\n")
        f.write("-"*70 + "\n")
        f.write(f"总文件数: {len(df)}\n")
        f.write(f"成功分析: {(df['status'] == 'success').sum()}\n")
        f.write(f"分析失败: {(df['status'] == 'error').sum()}\n\n")
        
        # 有效性统计
        if 'is_valid' in df.columns:
            f.write("结构有效性:\n")
            f.write("-"*70 + "\n")
            valid_count = df['is_valid'].sum()
            f.write(f"有效结构: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)\n")
            
            if 'sg_consistent' in df.columns:
                sg_count = df['sg_consistent'].sum()
                f.write(f"空间群一致: {sg_count}/{len(df)} ({sg_count/len(df)*100:.1f}%)\n")
            
            if 'multiplicity_consistent' in df.columns:
                mult_count = df['multiplicity_consistent'].sum()
                f.write(f"重数一致: {mult_count}/{len(df)} ({mult_count/len(df)*100:.1f}%)\n")
            
            if 'bond_length_score' in df.columns:
                mean_score = df['bond_length_score'].mean()
                f.write(f"平均键长合理性: {mean_score:.3f}\n")
            f.write("\n")
        
        # 质量分数分布
        if 'quality_score' in df.columns:
            f.write("质量分数分布:\n")
            f.write("-"*70 + "\n")
            f.write(f"平均质量分数: {df['quality_score'].mean():.1f}/100\n")
            f.write(f"中位数: {df['quality_score'].median():.1f}/100\n")
            f.write(f"最高分: {df['quality_score'].max():.1f}/100\n")
            f.write(f"最低分: {df['quality_score'].min():.1f}/100\n")
            f.write("\n")
            
            # 分数段统计
            excellent = (df['quality_score'] >= 80).sum()
            good = ((df['quality_score'] >= 60) & (df['quality_score'] < 80)).sum()
            fair = ((df['quality_score'] >= 40) & (df['quality_score'] < 60)).sum()
            poor = (df['quality_score'] < 40).sum()
            
            f.write(f"优秀 (≥80分): {excellent} ({excellent/len(df)*100:.1f}%)\n")
            f.write(f"良好 (60-79分): {good} ({good/len(df)*100:.1f}%)\n")
            f.write(f"一般 (40-59分): {fair} ({fair/len(df)*100:.1f}%)\n")
            f.write(f"较差 (<40分): {poor} ({poor/len(df)*100:.1f}%)\n")
            f.write("\n")
        
        # 容忍因子统计
        if 'tolerance_factor' in df.columns:
            df_with_tol = df[df['tolerance_factor'].notna()]
            if len(df_with_tol) > 0:
                f.write("Goldschmidt 容忍因子:\n")
                f.write("-"*70 + "\n")
                f.write(f"平均值: {df_with_tol['tolerance_factor'].mean():.3f}\n")
                f.write(f"中位数: {df_with_tol['tolerance_factor'].median():.3f}\n")
                f.write(f"范围: {df_with_tol['tolerance_factor'].min():.3f} - "
                       f"{df_with_tol['tolerance_factor'].max():.3f}\n")
                
                if 'tolerance_stability' in df.columns:
                    stable = (df['tolerance_stability'] == 'stable').sum()
                    marginal = (df['tolerance_stability'] == 'marginally_stable').sum()
                    unstable = (df['tolerance_stability'] == 'unstable').sum()
                    
                    f.write(f"\n稳定性评估:\n")
                    f.write(f"  稳定 (0.8-1.0): {stable}\n")
                    f.write(f"  边缘稳定 (0.7-0.8): {marginal}\n")
                    f.write(f"  不稳定 (<0.7 or >1.0): {unstable}\n")
                f.write("\n")
        
        # 晶系分布
        if 'crystal_system' in df.columns:
            f.write("晶系分布:\n")
            f.write("-"*70 + "\n")
            crystal_systems = df['crystal_system'].value_counts()
            for system, count in crystal_systems.items():
                f.write(f"{system}: {count} ({count/len(df)*100:.1f}%)\n")
            f.write("\n")
        
        # 前10个最高质量结构
        if 'quality_score' in df.columns:
            f.write("前10个最高质量结构:\n")
            f.write("-"*70 + "\n")
            top10 = df.nlargest(10, 'quality_score')
            
            for idx, row in top10.iterrows():
                f.write(f"\n{row.get('file', 'unknown')}:\n")
                f.write(f"  质量分数: {row['quality_score']:.0f}/100\n")
                if 'formula' in row:
                    f.write(f"  化学式: {row['formula']}\n")
                if 'space_group' in row:
                    f.write(f"  空间群: {row['space_group']}\n")
                if 'tolerance_factor' in row and pd.notna(row['tolerance_factor']):
                    f.write(f"  容忍因子: {row['tolerance_factor']:.3f}\n")
                if 'bond_length_score' in row:
                    f.write(f"  键长合理性: {row['bond_length_score']:.3f}\n")
    
    print(f"分析报告已保存到: {report_file}")
    
    # 筛选高质量结构
    if 'quality_score' in df.columns:
        high_quality = df[df['quality_score'] >= 80]
        if len(high_quality) > 0:
            hq_file = output_path / 'high_quality_structures.csv'
            high_quality.to_csv(hq_file, index=False)
            print(f"高质量结构（≥80分）已保存到: {hq_file}")
            print(f"  共 {len(high_quality)} 个结构")


def main():
    parser = argparse.ArgumentParser(
        description='分析和筛选钙钛矿结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析目录中的所有结构
  python analyze_perovskites.py --input perovskites/ --output analysis/
  
  # 只分析有效结构
  python analyze_perovskites.py --input perovskites/ --output analysis/ --valid-only
  
  # 设置质量阈值
  python analyze_perovskites.py --input perovskites/ --output analysis/ --min-quality 80
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='包含 CIF 文件的输入目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='analysis_results',
        help='输出目录（默认: analysis_results）'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归搜索子目录'
    )
    
    parser.add_argument(
        '--valid-only',
        action='store_true',
        help='只显示有效结构'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0,
        help='最小质量分数阈值（0-100）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("钙钛矿结构分析")
    print("="*70)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print("="*70 + "\n")
    
    # 分析结构
    df = analyze_directory(args.input, recursive=not args.no_recursive)
    
    if len(df) == 0:
        print("没有找到可分析的结构")
        return
    
    # 应用过滤器
    original_count = len(df)
    
    if args.valid_only:
        df = df[df['is_valid'] == True]
        print(f"\n过滤器: 仅有效结构 ({len(df)}/{original_count})")
    
    if args.min_quality > 0:
        df = df[df['quality_score'] >= args.min_quality]
        print(f"过滤器: 质量分数 ≥ {args.min_quality} ({len(df)}/{original_count})")
    
    if len(df) == 0:
        print("警告: 过滤后没有剩余结构")
        return
    
    # 生成报告
    generate_report(df, args.output)
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)


if __name__ == '__main__':
    main()
