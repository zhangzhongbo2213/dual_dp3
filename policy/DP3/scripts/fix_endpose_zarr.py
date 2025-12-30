"""
修复 DP3 EndPose 生成的 Zarr，其中 action 维度比 episode_ends[-1] 多 7 个时间步，
会触发 ReplayBuffer 的一致性断言。

做法：
- 读取原始 Zarr
- 取 episode_ends[-1] = N
- 将 action 截断为前 N 条，其他键保持不变
- 写回到新的 Zarr 目录（或覆盖原目录）
"""

import os
import shutil
import argparse
import numpy as np
import zarr


def fix_zarr(src_path: str, dst_path: str | None = None, overwrite: bool = False):
    src_path = os.path.abspath(src_path)
    if dst_path is None:
        # 默认在同目录下生成 *_fixed.zarr
        base, ext = os.path.splitext(src_path)
        if ext == ".zarr":
            dst_path = base + "_fixed.zarr"
        else:
            dst_path = src_path + "_fixed.zarr"
    dst_path = os.path.abspath(dst_path)

    if os.path.abspath(src_path) == os.path.abspath(dst_path) and not overwrite:
        raise ValueError("dst_path 与 src_path 相同，且未设置 --overwrite，避免误覆盖。")

    print(f"[fix_endpose_zarr] src: {src_path}")
    print(f"[fix_endpose_zarr] dst: {dst_path}")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源 Zarr 不存在: {src_path}")

    root = zarr.open(src_path, mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("Zarr 结构不完整，缺少 data 或 meta 分组。")

    data = root["data"]
    meta = root["meta"]

    if "episode_ends" not in meta:
        raise ValueError("meta 中缺少 episode_ends。")

    episode_ends = meta["episode_ends"][:]
    if episode_ends.size == 0:
        raise ValueError("episode_ends 为空。")

    final_t = int(episode_ends[-1])
    print(f"[fix_endpose_zarr] episode_ends[-1] = {final_t}")

    if "action" not in data:
        raise ValueError("data 中没有 action 键，不需要此脚本。")

    action = data["action"]
    print(f"[fix_endpose_zarr] 原 action.shape = {action.shape}")

    if action.shape[0] == final_t:
        print("[fix_endpose_zarr] action 长度与 episode_ends 一致，无需修复。")
        return

    if action.shape[0] < final_t:
        raise ValueError(
            f"action.shape[0] = {action.shape[0]} 小于 episode_ends[-1] = {final_t}，"
            "说明数据本身不完整，不能简单截断。"
        )

    # 准备目标目录
    if os.path.exists(dst_path):
        print(f"[fix_endpose_zarr] 目标路径已存在: {dst_path}，将删除后重建。")
        shutil.rmtree(dst_path)

    dst_store = zarr.DirectoryStore(dst_path)
    dst_root = zarr.group(store=dst_store)
    dst_meta = dst_root.create_group("meta")
    dst_data = dst_root.create_group("data")

    # 拷贝 meta
    for key, arr in meta.items():
        dst_meta.array(name=key, data=arr[:], chunks=arr.chunks, compressor=arr.compressor)

    # 拷贝 data，action 做截断
    for key, arr in data.items():
        if key == "action":
            new_arr = arr[:final_t]
            print(f"[fix_endpose_zarr] 截断 action 为前 {final_t} 条，shape={new_arr.shape}")
            dst_data.array(name=key, data=new_arr, chunks=arr.chunks, compressor=arr.compressor)
        else:
            dst_data.array(name=key, data=arr[:], chunks=arr.chunks, compressor=arr.compressor)

    print(f"[fix_endpose_zarr] 已写入修复后的 Zarr: {dst_path}")

    # 如需覆盖原路径，执行替换
    if overwrite and os.path.abspath(dst_path) != os.path.abspath(src_path):
        backup = src_path + ".bak"
        print(f"[fix_endpose_zarr] 覆盖原文件，先备份到: {backup}")
        if os.path.exists(backup):
            shutil.rmtree(backup)
        os.rename(src_path, backup)
        os.rename(dst_path, src_path)
        print(f"[fix_endpose_zarr] 覆盖完成，原文件已备份为: {backup}")


def main():
    parser = argparse.ArgumentParser(description="修复 DP3 EndPose Zarr 中 action 长度不一致的问题")
    parser.add_argument("src", type=str, help="源 Zarr 目录路径")
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="目标 Zarr 目录路径（默认在 src 同目录下生成 *_fixed.zarr）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖原始 Zarr（会先备份为 *.bak 再替换）",
    )
    args = parser.parse_args()

    fix_zarr(args.src, args.dst, args.overwrite)


if __name__ == "__main__":
    main()


