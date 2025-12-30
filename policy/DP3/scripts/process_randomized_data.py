#!/usr/bin/env python3
"""
统一处理 Randomized 数据
将 EndPose 和 GNN_EndPose 的输出保存在统一文件夹的不同子文件夹中
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# 获取脚本目录
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent.parent.parent  # /data/zzb/RoboTwin


def process_endpose_data(task_name, task_config, expert_data_num, output_base_dir):
    """处理 EndPose 数据"""
    print("\n" + "="*80)
    print("【步骤 1/2】处理 EndPose 数据")
    print("="*80)
    
    # 临时输出路径（使用原脚本的默认路径）
    temp_output = SCRIPT_DIR / "data" / f"{task_name}-{task_config}-{expert_data_num}-endpose.zarr"
    
    # 调用处理脚本
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "process_data_endpose.py"),
        task_name,
        task_config,
        str(expert_data_num)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    
    if result.returncode != 0:
        print(f"❌ EndPose 数据处理失败 (退出码: {result.returncode})")
        return False
    
    # 移动结果到统一目录
    final_output = output_base_dir / "endpose" / temp_output.name
    
    if temp_output.exists():
        print(f"移动数据: {temp_output} -> {final_output}")
        final_output.parent.mkdir(parents=True, exist_ok=True)
        if final_output.exists():
            shutil.rmtree(final_output)
        shutil.move(str(temp_output), str(final_output))
        print(f"✅ EndPose 数据已保存到: {final_output}")
        return True
    else:
        print(f"❌ 输出文件不存在: {temp_output}")
        return False


def process_gnn_endpose_data(task_name, task_config, expert_data_num, output_base_dir):
    """处理 GNN_EndPose 数据"""
    print("\n" + "="*80)
    print("【步骤 2/2】处理 GNN_EndPose 数据")
    print("="*80)
    
    # 临时输出路径（使用原脚本的默认路径）
    temp_output = SCRIPT_DIR / "data_gnn" / f"{task_name}-{task_config}-{expert_data_num}-gnn-endpose.zarr"
    
    # 调用处理脚本
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "process_data_gnn_endpose.py"),
        task_name,
        task_config,
        str(expert_data_num)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    
    if result.returncode != 0:
        print(f"❌ GNN_EndPose 数据处理失败 (退出码: {result.returncode})")
        return False
    
    # 移动结果到统一目录
    final_output = output_base_dir / "gnn_endpose" / temp_output.name
    
    if temp_output.exists():
        print(f"移动数据: {temp_output} -> {final_output}")
        final_output.parent.mkdir(parents=True, exist_ok=True)
        if final_output.exists():
            shutil.rmtree(final_output)
        shutil.move(str(temp_output), str(final_output))
        print(f"✅ GNN_EndPose 数据已保存到: {final_output}")
        return True
    else:
        print(f"❌ 输出文件不存在: {temp_output}")
        return False


def main():
    parser = argparse.ArgumentParser(description="统一处理 Randomized 数据")
    parser.add_argument("--task_name", type=str, default="stack_bowls_two",
                       help="任务名称 (默认: stack_bowls_two)")
    parser.add_argument("--task_config", type=str, default="demo_randomized",
                       help="任务配置 (默认: demo_randomized)")
    parser.add_argument("--expert_data_num", type=int, default=100,
                       help="数据量/episode数量 (默认: 100)")
    parser.add_argument("--output_base", type=str, default=None,
                       help="输出基础目录 (默认: ./data_processed/{task_name}/{task_config})")
    parser.add_argument("--skip_endpose", action="store_true",
                       help="跳过 EndPose 数据处理")
    parser.add_argument("--skip_gnn", action="store_true",
                       help="跳过 GNN_EndPose 数据处理")
    
    args = parser.parse_args()
    
    # 确定输出基础目录
    if args.output_base:
        output_base_dir = Path(args.output_base)
    else:
        output_base_dir = SCRIPT_DIR / "data_processed" / args.task_name / args.task_config
    
    print("="*80)
    print("处理 Randomized 数据")
    print("="*80)
    print(f"任务名称: {args.task_name}")
    print(f"任务配置: {args.task_config}")
    print(f"数据量: {args.expert_data_num} episodes")
    print(f"输出目录: {output_base_dir}")
    print("="*80)
    
    # 创建输出目录
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # 处理 EndPose 数据
    if not args.skip_endpose:
        if not process_endpose_data(args.task_name, args.task_config, args.expert_data_num, output_base_dir):
            success = False
    
    # 处理 GNN_EndPose 数据
    if not args.skip_gnn:
        if not process_gnn_endpose_data(args.task_name, args.task_config, args.expert_data_num, output_base_dir):
            success = False
    
    # 输出总结
    print("\n" + "="*80)
    if success:
        print("✅ 数据处理完成！")
        print("="*80)
        print("输出目录结构:")
        print(f"  {output_base_dir}/")
        if not args.skip_endpose:
            print(f"    ├── endpose/")
            print(f"    │   └── {args.task_name}-{args.task_config}-{args.expert_data_num}-endpose.zarr")
        if not args.skip_gnn:
            print(f"    └── gnn_endpose/")
            print(f"        └── {args.task_name}-{args.task_config}-{args.expert_data_num}-gnn-endpose.zarr")
    else:
        print("❌ 数据处理过程中出现错误")
    print("="*80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

