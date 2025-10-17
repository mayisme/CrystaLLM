#!/usr/bin/env python3
"""
示例脚本：批量生成钙钛矿结构

此脚本演示如何使用 CrystaLLM 批量生成多种钙钛矿组成的晶体结构。
可以指定化学组成、空间群和生成参数。

用法：
    python generate_perovskites.py --model crystallm_perov_5_small --output perovskites/
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymatgen.core import Composition
from crystallm import get_atomic_props_block_for_formula
import subprocess


# 常见钙钛矿组成和对应的空间群
COMMON_PEROVSKITES = {
    # 氧化物钙钛矿
    'CaTiO3': 'Pnma',      # 正交相
    'SrTiO3': 'Pm-3m',     # 立方相
    'BaTiO3': 'P4mm',      # 四方相（室温）
    'PbTiO3': 'P4mm',      # 四方相
    'LaAlO3': 'R-3c',      # 菱方相
    'KNbO3': 'Amm2',       # 正交相
    
    # 卤化物钙钛矿（用于太阳能电池）
    'CsPbI3': 'Pm-3m',     # 立方相（高温）
    'CsPbBr3': 'Pm-3m',    # 立方相
    'CsSnI3': 'Pnma',      # 正交相
    
    # 双钙钛矿
    'Sr2FeMoO6': 'I4/m',   # 四方相
    'Ba2NiWO6': 'Fm-3m',   # 立方相
}

# 钙钛矿空间群信息
PEROVSKITE_SPACE_GROUPS = {
    'cubic': ['Pm-3m', 'Fm-3m', 'Im-3'],
    'tetragonal': ['P4mm', 'P4/mmm', 'I4/mcm', 'I4/m'],
    'orthorhombic': ['Pnma', 'Pbnm', 'Cmcm', 'Amm2'],
    'rhombohedral': ['R-3c', 'R3c', 'R-3m'],
    'monoclinic': ['P21/m', 'C2/m'],
}


def create_prompt(composition, space_group=None):
    """
    创建用于生成的提示文本
    
    Args:
        composition: 化学组成字符串，如 'CaTiO3'
        space_group: 空间群符号，如 'Pm-3m'（可选）
    
    Returns:
        提示文本字符串
    """
    comp = Composition(composition)
    comp_str = comp.formula.replace(" ", "")
    
    if space_group:
        block = get_atomic_props_block_for_formula(comp_str)
        prompt = f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {space_group}\n"
        # 移除行首尾空格
        import re
        prompt = re.sub(r"^[ \t]+|[ \t]+$", "", prompt, flags=re.MULTILINE)
    else:
        prompt = f"data_{comp_str}\n"
    
    return prompt


def generate_structures(
    model_dir,
    compositions,
    output_dir,
    num_samples=10,
    temperature=0.75,
    top_k=10,
    device='cuda',
    max_tokens=3000
):
    """
    批量生成钙钛矿结构
    
    Args:
        model_dir: 训练好的模型目录
        compositions: 字典，键为组成，值为空间群
        output_dir: 输出目录
        num_samples: 每种组成生成的样本数
        temperature: 采样温度
        top_k: Top-K 采样参数
        device: 计算设备（'cuda' 或 'cpu'）
        max_tokens: 最大生成 token 数
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建临时目录存放提示文件
    prompts_dir = output_path / 'prompts'
    prompts_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for comp, space_group in compositions.items():
        print(f"\n{'='*60}")
        print(f"生成 {comp} 结构（空间群: {space_group or '未指定'}）")
        print(f"{'='*60}")
        
        # 创建提示文件
        prompt_text = create_prompt(comp, space_group)
        prompt_file = prompts_dir / f"{comp.replace('(', '').replace(')', '')}_prompt.txt"
        
        with open(prompt_file, 'w') as f:
            f.write(prompt_text)
        
        print(f"提示文件已创建: {prompt_file}")
        print(f"提示内容:\n{prompt_text[:200]}...")
        
        # 创建该组成的输出目录
        comp_output_dir = output_path / comp.replace('(', '').replace(')', '')
        comp_output_dir.mkdir(exist_ok=True)
        
        # 生成原始 CIF 文件到临时目录
        temp_dir = comp_output_dir / 'raw'
        temp_dir.mkdir(exist_ok=True)
        
        # 调用 sample.py
        cmd = [
            'python', str(project_root / 'bin' / 'sample.py'),
            f'out_dir={model_dir}',
            f'start=FILE:{prompt_file}',
            f'num_samples={num_samples}',
            f'max_new_tokens={max_tokens}',
            f'temperature={temperature}',
            f'top_k={top_k}',
            f'device={device}',
            'target=file',
        ]
        
        print(f"\n执行命令: {' '.join(cmd)}")
        
        # 在临时目录中生成
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"错误: {result.stderr}")
                results[comp] = {'status': 'failed', 'error': result.stderr}
            else:
                print(f"成功生成 {num_samples} 个样本")
                
                # 后处理
                processed_dir = comp_output_dir / 'processed'
                processed_dir.mkdir(exist_ok=True)
                
                postprocess_cmd = [
                    'python', str(project_root / 'bin' / 'postprocess.py'),
                    str(temp_dir),
                    str(processed_dir)
                ]
                
                print(f"后处理: {' '.join(postprocess_cmd)}")
                subprocess.run(postprocess_cmd, capture_output=True, text=True)
                
                results[comp] = {
                    'status': 'success',
                    'raw_dir': str(temp_dir),
                    'processed_dir': str(processed_dir),
                    'num_generated': num_samples
                }
                
                print(f"✓ {comp} 生成完成")
                print(f"  原始文件: {temp_dir}")
                print(f"  处理后文件: {processed_dir}")
                
        finally:
            os.chdir(original_dir)
    
    # 生成汇总报告
    report_file = output_path / 'generation_report.txt'
    with open(report_file, 'w') as f:
        f.write("钙钛矿结构生成报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型: {model_dir}\n")
        f.write(f"生成参数:\n")
        f.write(f"  - 每种组成样本数: {num_samples}\n")
        f.write(f"  - 温度: {temperature}\n")
        f.write(f"  - Top-K: {top_k}\n")
        f.write(f"  - 设备: {device}\n\n")
        
        f.write("生成结果:\n")
        f.write("-"*60 + "\n")
        
        for comp, result in results.items():
            f.write(f"\n{comp}:\n")
            if result['status'] == 'success':
                f.write(f"  状态: ✓ 成功\n")
                f.write(f"  生成数量: {result['num_generated']}\n")
                f.write(f"  处理后文件: {result['processed_dir']}\n")
            else:
                f.write(f"  状态: ✗ 失败\n")
                f.write(f"  错误: {result.get('error', 'Unknown')}\n")
    
    print(f"\n{'='*60}")
    print(f"所有生成任务完成！")
    print(f"报告已保存到: {report_file}")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='批量生成钙钛矿晶体结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成常见钙钛矿
  python generate_perovskites.py --model crystallm_perov_5_small --output perovskites/
  
  # 自定义组成
  python generate_perovskites.py --model crystallm_perov_5_small --output my_perovskites/ \\
      --compositions "CaTiO3:Pnma,SrTiO3:Pm-3m" --num-samples 20
  
  # 使用更保守的参数提高稳定性
  python generate_perovskites.py --model crystallm_perov_5_small --output stable/ \\
      --temperature 0.6 --top-k 5 --num-samples 50
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='训练好的模型目录路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='generated_perovskites',
        help='输出目录（默认: generated_perovskites）'
    )
    
    parser.add_argument(
        '--compositions', '-c',
        type=str,
        help='自定义组成列表，格式: "comp1:sg1,comp2:sg2"（如不指定则使用常见钙钛矿列表）'
    )
    
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=10,
        help='每种组成生成的样本数（默认: 10）'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.75,
        help='采样温度，较低值更保守（默认: 0.75）'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='Top-K 采样参数（默认: 10）'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备（默认: cuda）'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=3000,
        help='最大生成 token 数（默认: 3000）'
    )
    
    parser.add_argument(
        '--list-common',
        action='store_true',
        help='列出常见钙钛矿组成并退出'
    )
    
    args = parser.parse_args()
    
    # 列出常见钙钛矿
    if args.list_common:
        print("\n常见钙钛矿组成和空间群:\n")
        print(f"{'组成':<15} {'空间群':<10}")
        print("-" * 30)
        for comp, sg in COMMON_PEROVSKITES.items():
            print(f"{comp:<15} {sg:<10}")
        print("\n空间群类别:")
        for crystal_system, sgs in PEROVSKITE_SPACE_GROUPS.items():
            print(f"  {crystal_system}: {', '.join(sgs)}")
        return
    
    # 确定要生成的组成
    if args.compositions:
        # 解析自定义组成
        compositions = {}
        for item in args.compositions.split(','):
            if ':' in item:
                comp, sg = item.split(':')
                compositions[comp.strip()] = sg.strip()
            else:
                compositions[item.strip()] = None
    else:
        # 使用常见钙钛矿列表
        compositions = COMMON_PEROVSKITES
    
    print("\n" + "="*60)
    print("钙钛矿结构生成任务")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"输出目录: {args.output}")
    print(f"组成数量: {len(compositions)}")
    print(f"每种组成样本数: {args.num_samples}")
    print(f"总生成数量: {len(compositions) * args.num_samples}")
    print(f"温度: {args.temperature}")
    print(f"Top-K: {args.top_k}")
    print(f"设备: {args.device}")
    print("="*60 + "\n")
    
    # 确认继续
    try:
        response = input("继续？[Y/n] ")
        if response.lower() not in ['', 'y', 'yes']:
            print("已取消")
            return
    except KeyboardInterrupt:
        print("\n已取消")
        return
    
    # 生成结构
    results = generate_structures(
        model_dir=args.model,
        compositions=compositions,
        output_dir=args.output,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        max_tokens=args.max_tokens
    )
    
    # 统计
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"\n最终统计:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总数: {len(results)}")


if __name__ == '__main__':
    main()
