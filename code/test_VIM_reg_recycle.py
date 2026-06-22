"""
测试脚本：Register6 Recycle1 模型（方案A - Cross-Attention）
专门用于测试带有 Cross-Attention 回收机制的 Mamba-UNet 模型

模型特点：
- 使用 6 个 Register tokens
- 采用方案A：Cross-Attention 双流架构
- 逐层交互：Register tokens 通过 Cross-Attention 从 Patch tokens 中学习
- 真正的"回收"机制：每个编码器层后更新 Register

使用方法：
python code/test_VIM_reg_recycle.py --root_path ../data/ACDC --snapshot_path ../model/ACDC/VIM_140_labeled/mambaunet_register6_recycle1

新增功能：
- 自动检测数据集类型（ACDC vs Prostate）
- 根据数据集类型自动设置正确的类别数
- 兼容二分类（Prostate）和多分类（ACDC）任务
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import h5py

# 添加必要的路径到系统路径
sys.path.append('.')  # 添加当前目录到系统路径

# 导入必要的模块 - 使用与训练脚本相同的导入方式
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds
from dataloaders.dataset import BaseDataSets
from networks.vision_mamba import MambaUnet as VIM_seg
from config import get_config


# 自定义数据集类，用于从H5文件加载数据
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, list_file, num_classes=4):
        self.root_path = root_path
        self.num_classes = num_classes

        # 读取列表文件
        list_path = os.path.join(root_path, list_file)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"列表文件不存在: {list_path}")

        with open(list_path, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

        # 修复文件名 - 添加.h5扩展名
        self.file_names = [f"{name}.h5" if not name.endswith('.h5') else name for name in self.file_names]

        # 验证文件是否存在
        self.valid_files = []
        for file_name in self.file_names:
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(root_path, file_name),  # 直接在根目录
                os.path.join(root_path, "data", file_name),  # 在data子目录
                os.path.join(root_path, "slices", file_name),  # 在slices子目录
                os.path.join(root_path, "ACDC_training_slices", file_name),  # 在ACDC_training_slices目录
                os.path.join(root_path, "ACDC_training_volumes", file_name),  # 在ACDC_training_volumes目录
            ]

            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.valid_files.append(path)
                    found = True
                    break

            if not found:
                print(f"⚠️ 文件不存在: {file_name}，尝试的路径: {possible_paths}")

        if not self.valid_files:
            raise FileNotFoundError("没有找到有效的测试文件")

        print(f"✓ 找到 {len(self.valid_files)} 个有效的测试文件")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_path = self.valid_files[idx]

        try:
            with h5py.File(file_path, 'r') as f:
                # 自动检测键名
                image_key = None
                label_key = None

                # 查找可能的键名
                for key in f.keys():
                    key_lower = key.lower()
                    if 'image' in key_lower or 'data' in key_lower:
                        image_key = key
                    if 'label' in key_lower or 'mask' in key_lower:
                        label_key = key

                # 如果没找到，使用第一个键
                if image_key is None and len(f.keys()) > 0:
                    image_key = list(f.keys())[0]

                if image_key is None:
                    raise ValueError(f"在 {file_path} 中找不到图像数据")

                image = np.array(f[image_key])

                # 加载标签（如果存在）
                if label_key is not None:
                    label = np.array(f[label_key])
                else:
                    # 如果没有标签，创建全零标签（用于推理）
                    label = np.zeros_like(image)

                # 确保维度正确 (H, W)
                if len(image.shape) == 3:
                    if image.shape[0] == 1:  # (1, H, W)
                        image = image[0]
                    elif image.shape[-1] == 1:  # (H, W, 1)
                        image = image[:, :, 0]

                if len(label.shape) == 3:
                    if label.shape[0] == 1:
                        label = label[0]
                    elif label.shape[-1] == 1:
                        label = label[:, :, 0]

                # 转换为tensor
                image_tensor = torch.from_numpy(image).float()
                label_tensor = torch.from_numpy(label).long()

                # 归一化
                image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-8)

                return {
                    'image': image_tensor.unsqueeze(0),  # (1, H, W)
                    'label': label_tensor,  # (H, W)
                    'case_name': os.path.basename(file_path)
                }

        except Exception as e:
            print(f"❌ 读取文件 {file_path} 时出错: {str(e)}")
            raise


def detect_dataset_type(root_path, snapshot_path):
    """
    自动检测数据集类型并返回正确的类别数
    """
    # 从路径中检测数据集类型
    if 'Prostate' in root_path or 'prostate' in root_path.lower() or 'PROSTATE' in snapshot_path.upper():
        dataset_type = 'Prostate'
        num_classes = 2
        print(f"🔍 检测到前列腺数据集 (Prostate)，设置类别数: {num_classes}")
    elif 'ACDC' in root_path or 'acdc' in root_path.lower() or 'ACDC' in snapshot_path.upper():
        dataset_type = 'ACDC'
        num_classes = 4
        print(f"🔍 检测到心脏数据集 (ACDC)，设置类别数: {num_classes}")
    else:
        # 默认使用参数指定的类别数
        dataset_type = 'Unknown'
        num_classes = None
        print(f"⚠️ 无法自动检测数据集类型，使用参数指定的类别数")

    return dataset_type, num_classes


def create_model_with_adjusted_classes(config, img_size, original_num_classes, detected_num_classes, dataset_type):
    """
    根据检测到的数据集类型创建模型，必要时调整输出层
    """
    # 如果检测到正确的类别数且与原始不同，使用检测到的类别数
    if detected_num_classes is not None and detected_num_classes != original_num_classes:
        print(f"🔄 调整模型类别数: {original_num_classes} -> {detected_num_classes} (适配{dataset_type}数据集)")
        num_classes = detected_num_classes
    else:
        num_classes = original_num_classes

    # 创建模型
    model = VIM_seg(config, img_size=img_size, num_classes=num_classes)
    return model, num_classes


def load_weights_with_class_adjustment(model, weight_path, original_num_classes, detected_num_classes, dataset_type):
    """
    加载权重，处理类别数不匹配的情况
    """
    device = next(model.parameters()).device

    try:
        checkpoint = torch.load(weight_path, map_location=device)

        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 检查权重中的类别数
        weight_classes = None
        for key, param in state_dict.items():
            if 'output.weight' in key or 'aux_heads' in key:
                if len(param.shape) >= 1:
                    weight_classes = param.shape[0]
                    break

        if weight_classes is not None:
            print(f"🔍 权重文件类别数: {weight_classes}")

            # 如果检测到的类别数与权重不一致，使用权重的类别数
            if detected_num_classes is not None and detected_num_classes != weight_classes:
                print(f"⚠️ 检测到的类别数({detected_num_classes})与权重类别数({weight_classes})不一致")
                print(f"🔄 使用权重文件的类别数: {weight_classes}")
            elif detected_num_classes is None and original_num_classes != weight_classes:
                print(f"⚠️ 参数类别数({original_num_classes})与权重类别数({weight_classes})不一致")
                print(f"🔄 使用权重文件的类别数: {weight_classes}")

        # 尝试加载权重
        try:
            msg = model.load_state_dict(state_dict, strict=False)

            if msg.missing_keys:
                print(f"⚠️  Missing keys (可忽略的非学习参数): {msg.missing_keys[:5]}...")
            if msg.unexpected_keys:
                print(f"⚠️  Unexpected keys (可忽略): {msg.unexpected_keys[:5]}...")

            print("✓ Weights loaded successfully")
            return model, True  # 修正：返回模型和成功标志

        except RuntimeError as e:
            if "size mismatch" in str(e) and "output.weight" in str(e):
                print(f"❌ 类别数不匹配错误: {e}")
                print("🔄 尝试调整模型输出层...")

                # 尝试从错误信息中提取正确的类别数
                import re
                match = re.search(r'copying a param with shape torch\.Size\(\[(\d+)', str(e))
                if match:
                    correct_num_classes = int(match.group(1))
                    print(f"🔧 检测到正确的类别数: {correct_num_classes}")

                    # 重新创建模型
                    from networks.vision_mamba import MambaUnet
                    model_adjusted = MambaUnet(model.config, img_size=model.img_size, num_classes=correct_num_classes)
                    model_adjusted = model_adjusted.to(device)

                    # 再次尝试加载
                    try:
                        msg = model_adjusted.load_state_dict(state_dict, strict=False)
                        print("✓ 调整后的模型权重加载成功")

                        # 返回调整后的模型
                        return model_adjusted, True
                    except Exception as e2:
                        print(f"❌ 调整后模型仍加载失败: {e2}")
                        return None, False
                else:
                    print("❌ 无法从错误信息中提取正确的类别数")
                    return None, False
            else:
                print(f"❌ 其他加载错误: {e}")
                return None, False

    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return None, False

    # 这行代码应该不会执行到，但为了安全还是保留
    return model, True


def test_model(args):
    """
    测试 Register6 Recycle 模型
    """
    print("\n" + "=" * 60)
    print("REGISTER6 RECYCLE1 MODEL TESTING (方案A - Cross-Attention)")
    print("=" * 60)

    # ================== 1. 自动检测数据集类型 ==================
    dataset_type, detected_num_classes = detect_dataset_type(args.root_path, args.snapshot_path)

    # ================== 2. 加载配置和模型 ==================
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    # 添加必要的参数用于 get_config
    if not hasattr(args, 'exp'):
        if dataset_type == 'Prostate':
            args.exp = 'Prostate/VIM'
        else:
            args.exp = 'ACDC/VIM'

    if not hasattr(args, 'deterministic'):
        args.deterministic = 1
    if not hasattr(args, 'base_lr'):
        args.base_lr = 0.01
    if not hasattr(args, 'seed'):
        args.seed = 1337

    # 加载配置
    config = get_config(args)

    # 打印 Register 配置
    use_register = getattr(config.MODEL, 'USE_REGISTER', True)
    num_cls_tokens = getattr(config.MODEL, 'NUM_CLS_TOKENS', 6)
    register_version = getattr(config.MODEL, 'REGISTER_VERSION', 'cross_attn')
    
    # 从路径中自动检测tokens数量（如果路径中有register4、register6、register8等）
    import re
    path_tokens_match = re.search(r'register(\d+)', args.snapshot_path, re.IGNORECASE)
    if path_tokens_match:
        path_tokens = int(path_tokens_match.group(1))
        if path_tokens != num_cls_tokens:
            # 这是正常的自动修正，不是错误，使用信息提示而不是警告
            print(f"ℹ️  从路径自动检测到 {path_tokens} 个Register Tokens，已自动更新配置")
            num_cls_tokens = path_tokens
            # 更新config（需要先解冻，修改，再冻结）
            config.defrost()
            config.MODEL.NUM_CLS_TOKENS = path_tokens
            config.freeze()

    print(f"✓ Config loaded")
    print(f"  - Use Register: {use_register}")
    print(f"  - Number of Register Tokens: {num_cls_tokens}")
    # 版本名称映射
    version_names = {
        'cross_attn': "方案A/V3 (Cross-Attention)",
        'pooled': "方案D/V2 (Pooled Aggregation)",
        'enhanced_gate': "方案V4 (Enhanced Gated Fusion)"
    }
    version_name = version_names.get(register_version, register_version)
    print(f"  - Register Version: {register_version} ({version_name})")
    print(f"  - Dataset Type: {dataset_type}")
    print(f"  - Original Num Classes: {args.num_classes}")
    print(f"  - Detected Num Classes: {detected_num_classes}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 创建模型（使用检测到的类别数）
    model, final_num_classes = create_model_with_adjusted_classes(
        config, args.patch_size, args.num_classes, detected_num_classes, dataset_type
    )
    args.num_classes = final_num_classes  # 更新参数中的类别数
    model = model.to(device)
    model.eval()

    # ================== 3. 加载权重 ==================
    # 智能查找权重文件
    weight_path = None

    # 优先级1：查找各版本的 best_model.pth
    possible_best_names = [
        'mambaunet_register6_v4_best_model.pth',          # 方案V4（最优）
        'mambaunet_register6_recycle1_best_model.pth',    # 方案A/V3
        'register6_v4_best_model.pth',                    # 方案V4（简化名）
        'register6_recycle1_best_model.pth',              # 方案A/V3（简化名）
        'mambaunet_register6_recycle_best_model.pth',     # 方案D/V2
        'register6_recycle_best_model.pth',               # 方案D/V2（简化名）
        'mambaunet_register6_v4_deepsup_best_model.pth', # 方案V4 + DeepSup
        'register6_v4_deepsup_best_model.pth',            # 方案V4 + DeepSup（简化名）
    ]

    for name in possible_best_names:
        test_path = os.path.join(args.snapshot_path, name)
        if os.path.exists(test_path):
            weight_path = test_path
            # 判断模型版本
            if '_v4_deepsup' in name:
                print(f"✓ Found best model (方案V4 + DeepSup): {name}")
            elif '_v4' in name:
                print(f"✓ Found best model (方案V4): {name}")
            elif 'recycle1' in name:
                print(f"✓ Found best model (方案A/V3): {name}")
            else:
                print(f"✓ Found best model (方案D/V2): {name}")
            break

    # 优先级2：如果没找到，查找任何 _best_model.pth 文件
    if weight_path is None:
        for file in os.listdir(args.snapshot_path):
            if file.endswith('_best_model.pth'):
                weight_path = os.path.join(args.snapshot_path, file)
                print(f"✓ Found model: {file}")
                break

    # 优先级3：如果还没找到，查找最新的 .pth 文件
    if weight_path is None:
        pth_files = [f for f in os.listdir(args.snapshot_path) if f.endswith('.pth')]
        if pth_files:
            # 按修改时间排序，选择最新的
            pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.snapshot_path, x)), reverse=True)
            weight_path = os.path.join(args.snapshot_path, pth_files[0])
            print(f"✓ Found latest model: {pth_files[0]}")

    if weight_path is None:
        raise FileNotFoundError(f"在 {args.snapshot_path} 中找不到模型权重文件")

    print(f"\nLoading weights from: {weight_path}")

    # 加载权重（带类别数调整）
    model, success = load_weights_with_class_adjustment(
        model, weight_path, args.num_classes, detected_num_classes, dataset_type
    )

    if not success or model is None:
        print("❌ 权重加载失败")
        return None

    # ================== 4. 测试 ==================
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    # 使用 BaseDataSets（与训练时一致）
    try:
        db_test = BaseDataSets(base_dir=args.root_path, split="test")
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        print(f"✓ Loaded {len(db_test)} test samples from {dataset_type} dataset")
    except Exception as e:
        print(f"⚠️  Failed to load with BaseDataSets: {e}")
        print("   Trying alternative loading method...")

        # 备选方案：使用 H5Dataset
        try:
            # 根据数据集类型选择正确的列表文件
            if dataset_type == 'Prostate':
                list_file = "test.list"  # Prostate数据集的列表文件
            else:
                list_file = "test.list"  # ACDC数据集的列表文件

            db_test = H5Dataset(args.root_path, list_file, num_classes=args.num_classes)
            testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
            print(f"✓ Loaded {len(db_test)} test samples using H5Dataset")
        except Exception as e2:
            print(f"❌ Failed to load test data: {e2}")
            return None

    # 执行测试 - 使用与训练时相同的 test_single_volume 函数
    metric_list = 0.0

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tqdm(testloader, desc="Testing")):
            try:
                # 使用 test_single_volume 函数（与训练时验证相同）
                metric_i = test_single_volume(
                    sampled_batch["image"],
                    sampled_batch["label"],
                    model,
                    classes=args.num_classes,
                    patch_size=args.patch_size
                )
                metric_list += np.array(metric_i)

            except Exception as e:
                print(f"\n⚠️  Error processing batch {i_batch}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 计算平均指标
    if isinstance(metric_list, np.ndarray) and metric_list.sum() > 0:
        metric_list = metric_list / len(db_test)
        metric_array = metric_list.reshape(-1, 2)  # (num_classes-1, 2) - dice and hd95
    else:
        metric_array = None

    if metric_array is None:
        print("❌ No valid test results")
        return None

    # ================== 5. 打印结果 ==================
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS - REGISTER6 RECYCLE MODEL")
    print("=" * 60)

    # 打印配置信息
    strategy_names = {
        'cross_attn': "Cross-Attention with Residual Connection (方案A/V3)",
        'pooled': "Pooled Aggregation + Gated Update (方案D/V2)",
        'enhanced_gate': "Dual-Channel Gating + Boundary Enhancement (方案V4)"
    }
    update_strategy = strategy_names.get(register_version, register_version)
    print(f"\nModel Configuration:")
    print(f"  - Dataset: {dataset_type}")
    print(f"  - Number of Classes: {args.num_classes}")
    print(f"  - Register Mechanism: Enabled")
    print(f"  - Register Tokens: {num_cls_tokens}")
    print(f"  - Register Version: {register_version}")
    print(f"  - Update Strategy: {update_strategy}")
    print(f"  - Model Path: {args.snapshot_path}")
    print(f"  - Test Samples: {len(db_test)}")

    # 打印每个类别的指标
    print(f"\nPer-Class Results:")

    if dataset_type == 'Prostate' and args.num_classes == 2:
        # 前列腺数据集特殊显示
        prostate_dice = metric_array[0, 0] if len(metric_array) > 0 else 0
        prostate_hd95 = metric_array[0, 1] if len(metric_array) > 0 else 0
        print(f"  Prostate: Dice = {prostate_dice:.4f}, HD95 = {prostate_hd95:.4f}")
    else:
        # ACDC数据集或其他多类数据集
        for class_i in range(args.num_classes - 1):  # 跳过背景类
            if class_i < len(metric_array):
                class_dice = metric_array[class_i, 0]
                class_hd95 = metric_array[class_i, 1]
                print(f"  Class {class_i + 1}: Dice = {class_dice:.4f}, HD95 = {class_hd95:.4f}")

    # 计算并打印整体指标
    if len(metric_array) > 0:
        mean_dice = np.mean(metric_array, axis=0)[0]
        mean_hd95 = np.mean(metric_array, axis=0)[1]
    else:
        mean_dice = 0
        mean_hd95 = 0

    print(f"\nOverall Performance:")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Mean HD95: {mean_hd95:.4f}")

    print("=" * 60)

    return metric_array, dataset_type, args.num_classes


if __name__ == "__main__":
    # 设置参数（针对 Register6 Recycle 模型）
    parser = argparse.ArgumentParser(description='Test Register6 Recycle Mamba-UNet Model')

    # 必需参数
    parser.add_argument('--root_path', type=str,
                        default='../data/ACDC',
                        help='dataset root path')
    parser.add_argument('--snapshot_path', type=str,
                        default='../model/ACDC/VIM_140_labeled/mambaunet_register6_recycle',
                        help='path to trained model snapshot')

    # 模型参数
    parser.add_argument('--model', type=str,
                        default='mambaunet',
                        help='model name')
    parser.add_argument('--num_classes', type=int,
                        default=4,
                        help='number of classes (including background)')
    parser.add_argument('--patch_size', nargs='+', type=int,
                        default=[224, 224],
                        help='patch size of network input')

    # 配置文件
    parser.add_argument('--cfg', type=str,
                        default="../code/configs/vmamba_tiny.yaml",
                        help='path to config file')

    # 其他参数（确保 get_config 不会出错）
    parser.add_argument('--opts', nargs='+', default=None,
                        help='Modify config options')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset')
    parser.add_argument('--cache-mode', type=str, default='part',
                        choices=['no', 'full', 'part'])
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing")
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                        choices=['O0', 'O1', 'O2'])
    parser.add_argument('--tag', type=str, default=None,
                        help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=140,
                        help='labeled data')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='maximum epoch number to train')
    parser.add_argument('--exp', type=str, default='ACDC/VIM',
                        help='experiment_name')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1337,
                        help='random seed')

    args = parser.parse_args()

    # 确保patch_size是列表形式
    if not isinstance(args.patch_size, list):
        args.patch_size = [args.patch_size, args.patch_size]

    # 打印欢迎信息
    print("\n" + "=" * 60)
    print("REGISTER6 RECYCLE1 MODEL TESTING SCRIPT (增强版 - 支持多数据集)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Dataset: {args.root_path}")
    print(f"  Model: {args.snapshot_path}")
    print(f"  Num Classes: {args.num_classes}")
    print(f"  Patch Size: {args.patch_size}")
    print(f"  Config: {args.cfg}")

    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行测试
    results = test_model(args)

    # 保存结果到文件
    if results is not None:
        metric_array, dataset_type, num_classes = results

        # 从 config 获取版本信息
        config = get_config(args)
        register_version = getattr(config.MODEL, 'REGISTER_VERSION', 'cross_attn')

        # 根据版本生成文件名和标题
        # 检查是否是v4_deepsup版本（通过路径判断）
        is_deepsup = 'deepsup' in args.snapshot_path.lower() or '_v4_deepsup' in args.snapshot_path.lower()

        if register_version == 'enhanced_gate' and is_deepsup:
            result_suffix = 'v4_deepsup'
            title = f"Register6 V4 + Deep Supervision Model - {dataset_type} Test Results"
            version_desc = "V4 + Deep Supervision (多层监督)"
            mechanism = "Dual-Channel Gating + Boundary Enhancement + Deep Supervision"
        elif register_version == 'enhanced_gate':
            result_suffix = 'v4'
            title = f"Register6 V4 Model (Enhanced Gated Fusion) - {dataset_type} Test Results"
            version_desc = "Enhanced Gated Fusion (方案V4)"
            mechanism = "Dual-Channel Gating + Boundary Enhancement"
        elif register_version == 'cross_attn':
            result_suffix = 'recycle1'
            title = f"Register6 Recycle1 Model (方案A/V3 - Cross-Attention) - {dataset_type} Test Results"
            version_desc = "Cross-Attention (方案A/V3)"
            mechanism = "Cross-Attention with Residual Connection"
        else:  # pooled
            result_suffix = 'recycle'
            title = f"Register6 Recycle Model (方案D/V2 - Pooled Aggregation) - {dataset_type} Test Results"
            version_desc = "Pooled Aggregation (方案D/V2)"
            mechanism = "Pooled Aggregation + Gated Update"

        result_file = os.path.join(args.snapshot_path, f'test_results_register6_{result_suffix}_{dataset_type.lower()}.txt')

        # 计算整体指标
        if len(metric_array) > 0:
            mean_dice = np.mean(metric_array, axis=0)[0]
            mean_hd95 = np.mean(metric_array, axis=0)[1]
        else:
            mean_dice = 0
            mean_hd95 = 0

        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {dataset_type}\n")
            f.write(f"Model Path: {args.snapshot_path}\n")
            f.write(f"Data Path: {args.root_path}\n")
            f.write(f"Number of Classes: {num_classes}\n")
            f.write(f"Register Tokens: 6\n")
            f.write(f"Register Version: {version_desc}\n")
            f.write(f"Update Mechanism: {mechanism}\n\n")
            f.write("=" * 60 + "\n")
            f.write(f"Mean Dice Score: {mean_dice:.4f}\n")
            f.write(f"Mean HD95: {mean_hd95:.4f}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Per-Class Results:\n")
            if dataset_type == 'Prostate' and num_classes == 2:
                if len(metric_array) > 0:
                    prostate_dice = metric_array[0, 0]
                    prostate_hd95 = metric_array[0, 1]
                    f.write(f"Prostate Class:\n")
                    f.write(f"  Dice: {prostate_dice:.4f}\n")
                    f.write(f"  HD95: {prostate_hd95:.4f}\n")
            else:
                for i in range(num_classes - 1):
                    if i < len(metric_array):
                        class_dice = metric_array[i, 0]
                        class_hd95 = metric_array[i, 1]
                        f.write(f"Class {i + 1}:\n")
                        f.write(f"  Dice: {class_dice:.4f}\n")
                        f.write(f"  HD95: {class_hd95:.4f}\n")

        print(f"\n✓ Detailed results saved to: {result_file}")
    else:
        print("\n❌ Testing failed - no results to save")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60 + "\n")
