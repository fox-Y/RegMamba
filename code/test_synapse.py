"""
Synapse数据集测试脚本
用于测试在Synapse数据集上训练的模型

使用方法:
python test_synapse.py \
    --root_path ../data/Synapse \
    --exp Synapse/VIM_18_labeled \
    --model mambaunet \
    --num_classes 9 \
    --model_path ../model/Synapse/VIM_18_labeled/mambaunet_register6_v4_deepsup/mambaunet_register6_v4_deepsup_best_model.pth
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import h5py
from scipy.ndimage import zoom
from medpy import metric

# 添加路径
sys.path.append('.')

from dataloaders.dataset import BaseDataSets_Synapse
from networks.vision_mamba import MambaUnet as VIM_seg
from config import get_config


def calculate_metric_percase(pred, gt):
    """计算单个类别的Dice和HD95"""
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0  # 两者都是0，Dice=1
    else:
        return 0.0, 0.0


def test_single_volume(image, label, net, classes, patch_size=[224, 224], verbose=False):
    """
    测试单个3D体积
    
    Args:
        image: (1, H, W, D) - 3D图像
        label: (H, W, D) - 3D标签
        net: 模型
        classes: 类别数（包括背景）
        patch_size: 输入patch大小
        verbose: 是否显示详细进度
    """
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # 如果是2D，添加一个维度
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]
    
    prediction = np.zeros_like(label)
    num_slices = image.shape[0]
    
    # 逐切片处理
    for ind in range(num_slices):
        if verbose and ind % 10 == 0:
            print(f"  处理切片 {ind+1}/{num_slices}...", end='\r')
        
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        
        # 调整到patch_size
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            output = net(input)
            # 处理Deep Supervision的输出
            if isinstance(output, tuple):
                output = output[0]  # 只使用主输出
            
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            
            # 调整回原始尺寸
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    
    if verbose:
        print(f"  完成 {num_slices} 个切片的处理")
    
    # 计算每个类别的指标
    metric_list = []
    for i in range(1, classes):  # 跳过背景（类别0）
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    
    return metric_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse', help='数据根目录')
    parser.add_argument('--exp', type=str,
                        default='Synapse/VIM_18_labeled', help='实验名称')
    parser.add_argument('--model', type=str,
                        default='mambaunet', help='模型名称')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='类别数（包括背景）')
    parser.add_argument('--model_path', type=str,
                        default='', help='模型权重路径')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224],
                        help='输入patch大小 (height width)')
    parser.add_argument('--cfg', type=str, default='configs/vmamba_tiny.yaml',
                        help='配置文件路径')
    parser.add_argument('--opts', nargs='+', default=None,
                        help='其他配置选项')
    parser.add_argument('--config_name', type=str, default=None,
                        help='配置名称 (config1/config2/config3)，用于自动设置路径和tokens')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args)
    
    # 从路径或参数中自动检测tokens数量和配置路径
    import re
    num_cls_tokens = 6  # 默认值
    config_name = None
    
    # 优先从config_name参数获取
    if args.config_name:
        config_name = args.config_name.lower()
    # 其次从exp名称中检测
    elif args.exp:
        exp_lower = args.exp.lower()
        exp_config_match = re.search(r'config(\d+)', exp_lower, re.IGNORECASE)
        if exp_config_match:
            config_name = f"config{exp_config_match.group(1)}"
        elif 'configc' in exp_lower:
            config_name = "configc"
    # 最后从model_path中检测
    elif args.model_path:
        path_lower = args.model_path.lower()
        path_config_match = re.search(r'config(\d+)', path_lower, re.IGNORECASE)
        if path_config_match:
            config_name = f"config{path_config_match.group(1)}"
        elif 'configc' in path_lower:
            config_name = "configc"
    
    # 根据config名称确定tokens数量
    if config_name:
        tokens_map = {"config1": 8, "config2": 10, "config3": 12, "configc": 12}
        if config_name in tokens_map:
            num_cls_tokens = tokens_map[config_name]
            print(f"ℹ️  检测到{config_name}，使用 {num_cls_tokens} 个Register Tokens")
            config.defrost()
            config.MODEL.NUM_CLS_TOKENS = num_cls_tokens
            config.freeze()
    
    # 从model_path中检测tokens数量（如果还没设置）
    if args.model_path and num_cls_tokens == 6:
        path_tokens_match = re.search(r'register(\d+)', args.model_path, re.IGNORECASE)
        if path_tokens_match:
            num_cls_tokens = int(path_tokens_match.group(1))
            print(f"ℹ️  从路径自动检测到 {num_cls_tokens} 个Register Tokens")
            config.defrost()
            config.MODEL.NUM_CLS_TOKENS = num_cls_tokens
            config.freeze()
    
    # 设置模型保存路径
    if args.model_path == '':
        # 根据config名称或exp名称自动构建路径
        exp_lower = args.exp.lower()
        if config_name == "config1" or 'config1' in exp_lower:
            snapshot_path = f"../model/Synapse/VIM_18_labeled_config1_18_labeled/mambaunet_register8_v4_deepsup"
            model_path = os.path.join(snapshot_path, "mambaunet_register8_v4_deepsup_best_model.pth")
        elif config_name == "config2" or 'config2' in exp_lower:
            snapshot_path = f"../model/Synapse/VIM_18_labeled_config2_18_labeled/mambaunet_register10_v4_deepsup"
            model_path = os.path.join(snapshot_path, "mambaunet_register10_v4_deepsup_best_model.pth")
        elif config_name == "config3" or 'config3' in exp_lower:
            snapshot_path = f"../model/Synapse/VIM_18_labeled_config3_18_labeled/mambaunet_register12_v4_deepsup"
            model_path = os.path.join(snapshot_path, "mambaunet_register12_v4_deepsup_best_model.pth")
        elif config_name == "configc" or 'configc' in exp_lower:
            snapshot_path = f"../model/Synapse/VIM_18_labeled_configC_18_labeled/mambaunet_register12_v4_deepsup"
            model_path = os.path.join(snapshot_path, "mambaunet_register12_v4_deepsup_best_model.pth")
        else:
            # 默认路径
            snapshot_path = f"../model/{args.exp}/{args.model}_register6_v4_deepsup"
            model_path = os.path.join(snapshot_path, f"{args.model}_register6_v4_deepsup_best_model.pth")
    else:
        model_path = args.model_path
        snapshot_path = os.path.dirname(model_path)
    
    print("=" * 60)
    print("Synapse数据集测试")
    print("=" * 60)
    print(f"数据路径: {args.root_path}")
    print(f"模型路径: {model_path}")
    print(f"类别数: {args.num_classes}")
    print(f"Patch大小: {args.patch_size}")
    print("=" * 60)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("请检查模型路径是否正确，或先训练模型。")
        return
    
    # 加载模型
    print("📦 加载模型...")
    model = VIM_seg(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    
    # 加载权重
    try:
        checkpoint = torch.load(model_path, map_location='cuda')
        
        # 检查是否是完整checkpoint（包含model_state_dict）还是纯模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整checkpoint，提取模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 成功加载模型（从完整checkpoint）: {model_path}")
            if 'best_performance' in checkpoint:
                print(f"   最佳性能: Dice={checkpoint.get('best_performance', 'N/A'):.4f}, "
                      f"HD95={checkpoint.get('mean_hd95', 'N/A'):.4f}")
        else:
            # 纯模型权重
            model.load_state_dict(checkpoint)
            print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model.eval()
    
    # 加载测试数据
    print("📂 加载测试数据...")
    try:
        db_test = BaseDataSets_Synapse(base_dir=args.root_path, split="val")
        print(f"✅ 数据加载完成，测试样本数: {len(db_test)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)  # 改为0避免多进程问题
    
    # 测试
    print("\n🧪 开始测试...")
    print(f"将处理 {len(db_test)} 个测试样本")
    metric_list = 0.0
    
    # Synapse的8个器官名称
    organ_names = [
        "Aorta", "Gallbladder", "Spleen", "Left Kidney",
        "Right Kidney", "Liver", "Pancreas", "Stomach"
    ]
    
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tqdm(testloader, desc="测试中")):
            image, label = sampled_batch['image'], sampled_batch['label']
            image, label = image.cuda(), label.cuda()
            
            metric_i = test_single_volume(
                image, label, model, 
                classes=args.num_classes, 
                patch_size=args.patch_size
            )
            metric_list += np.array(metric_i)
    
    # 计算平均指标
    metric_list = metric_list / len(db_test)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    
    # 每个器官的结果
    print("\n每个器官的Dice和HD95:")
    print("-" * 60)
    print(f"{'器官':<20} {'Dice':<10} {'HD95':<10}")
    print("-" * 60)
    
    for i, organ_name in enumerate(organ_names):
        dice = metric_list[i, 0]
        hd95 = metric_list[i, 1]
        print(f"{organ_name:<20} {dice:<10.4f} {hd95:<10.4f}")
    
    # 平均结果
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    print("-" * 60)
    print(f"{'平均':<20} {mean_dice:<10.4f} {mean_hd95:<10.4f}")
    print("=" * 60)
    
    # 保存结果到文件
    result_file = os.path.join(snapshot_path, "test_results_synapse.txt")
    with open(result_file, 'w') as f:
        f.write("Synapse数据集测试结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"数据路径: {args.root_path}\n")
        f.write(f"类别数: {args.num_classes}\n")
        f.write(f"测试样本数: {len(db_test)}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("每个器官的结果:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'器官':<20} {'Dice':<10} {'HD95':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, organ_name in enumerate(organ_names):
            dice = metric_list[i, 0]
            hd95 = metric_list[i, 1]
            f.write(f"{organ_name:<20} {dice:<10.4f} {hd95:<10.4f}\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'平均':<20} {mean_dice:<10.4f} {mean_hd95:<10.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n✅ 结果已保存到: {result_file}")
    print("\n🎉 测试完成！")


if __name__ == '__main__':
    main()

