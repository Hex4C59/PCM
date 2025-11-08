#!/usr/bin/env python
# 保存路径示例：scripts/preprocess_label/split_audio_dataset.py
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple

# 根据需要调整映射：Session -> split
SESSION_TO_SPLIT: Dict[str, str] = {
    "Session1": "train",
    "Session2": "train",
    "Session3": "train",
    "Session4": "val",
    "Session5": "test",  # 如果要放到 test，把值改成 "test"
}

def reorganize_dataset(src: Path, dst: Path, move: bool, dry_run: bool) -> Dict[str, int]:
    """按照 SESSION_TO_SPLIT 重排 Session 目录"""
    if not src.exists():
        raise FileNotFoundError(f"源目录不存在：{src}")
    dst.mkdir(parents=True, exist_ok=True)

    stats = {split: 0 for split in set(SESSION_TO_SPLIT.values())}
    for session_dir in sorted(src.iterdir()):
        if not session_dir.is_dir():
            continue
        split = SESSION_TO_SPLIT.get(session_dir.name)
        if split is None:
            print(f"[跳过] 未在映射中找到 {session_dir.name}")
            continue

        target_split_root = dst / split
        if not dry_run:
            target_split_root.mkdir(parents=True, exist_ok=True)

        for file_path in session_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(session_dir)
            target_file = target_split_root / rel_path
            if not dry_run:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(file_path), str(target_file))
                else:
                    shutil.copy2(file_path, target_file)
            stats[split] = stats.get(split, 0) + 1
        print(f"[完成] {session_dir.name} -> {split}")

    return stats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 IEMOCAP Session 目录重新划分为 train/val/test"
    )
    parser.add_argument("--src", default="data/raw/iemocap_audio", type=Path,
                        help="原始 Session 根目录")
    parser.add_argument("--dst", default="data/raw/iemocap_audio_split", type=Path,
                        help="输出的划分根目录")
    parser.add_argument("--move", action="store_true",
                        help="使用移动替代复制，节省空间但会改动原目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="只输出统计信息，不实际复制/移动")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    stats = reorganize_dataset(args.src, args.dst, args.move, args.dry_run)
    for split, count in stats.items():
        print(f"{split}: {count} 个文件")
    if args.dry_run:
        print("dry-run 模式下未对文件做任何修改")

if __name__ == "__main__":
    main()
