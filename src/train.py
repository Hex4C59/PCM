#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCM项目 - 主训练入口
==================

统一训练入口，支持多种模式：
1. 预训练模型特征提取 (推荐)
2. 预训练模型训练
3. 基础模型训练
4. 多任务训练
5. 测试预训练模型系统

作者：PCM项目团队
日期：2025年
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append('/mnt/shareEEx/liuyang/code/PCM')

def main():
    """主训练入口"""
    print("="*60)
    print("🎯 PCM项目主入口")
    print("="*60)
    print("请选择模式:")
    print("1. 预训练模型特征提取 (推荐)")
    print("2. 预训练模型训练")
    print("3. 基础模型训练")
    print("4. 多任务训练")
    print("5. 测试预训练模型系统")
    print("="*60)

    while True:
        try:
            choice = input("请输入选项 (1-5) [默认: 1]: ").strip()
            if not choice:
                choice = "1"

            if choice == "1":
                print("\n🚀 启动预训练模型特征提取...")
                from scripts.preprocess_features.pretrained_feature_extractor import main as feature_main
                feature_main()
                break

            elif choice == "2":
                print("\n🚀 启动预训练模型训练...")
                from scripts.train_pretrained_model import main as pretrained_main
                pretrained_main()
                break

            elif choice == "3":
                print("\n🚀 启动基础模型训练...")
                from data.english_ver.iemocap_audio_train import main as basic_main
                basic_main()
                break

            elif choice == "4":
                print("\n🚀 启动多任务训练...")
                # 这里可以添加多任务训练的入口
                print("多任务训练功能开发中...")
                break

            elif choice == "5":
                print("\n🧪 启动预训练模型系统测试...")
                from test_pretrained_model import main as test_main
                test_main()
                break

            else:
                print("❌ 无效选项，请输入 1-5")

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断操作")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()