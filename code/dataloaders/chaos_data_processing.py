"""
CHAOS数据集预处理脚本
将DICOM图像和PNG标注转换为H5格式，类似Synapse数据集

使用方法:
python chaos_data_processing.py --data_dir ../data/CHAOS --output_dir ../data/CHAOS --modality MR
"""

import os
import numpy as np
import h5py
import cv2
from PIL import Image
import pydicom
from tqdm import tqdm
import argparse
from glob import glob
import re


def load_dicom_series(dicom_dir, use_inphase=True):
    """
    加载DICOM序列，按切片位置排序
    
    Args:
        dicom_dir: DICOM文件目录
        use_inphase: 对于MR T1DUAL，是否使用InPhase（True）或OutPhase（False）
    
    Returns:
        volume: (H, W, D) numpy数组
    """
    # 检查是否有InPhase和OutPhase子目录（MR T1DUAL）
    inphase_dir = os.path.join(dicom_dir, 'InPhase')
    outphase_dir = os.path.join(dicom_dir, 'OutPhase')
    
    if os.path.exists(inphase_dir) and os.path.exists(outphase_dir):
        # MR T1DUAL数据：选择InPhase或OutPhase
        if use_inphase:
            dicom_dir = inphase_dir
        else:
            dicom_dir = outphase_dir
    
    dicom_files = glob(os.path.join(dicom_dir, '*.dcm'))
    
    if len(dicom_files) == 0:
        print(f"Warning: No DICOM files found in {dicom_dir}")
        return None
    
    # 读取所有DICOM文件并获取切片位置
    slices = []
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file)
            # 获取像素数据
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # 获取切片位置（如果存在）
            if hasattr(ds, 'SliceLocation'):
                slice_location = ds.SliceLocation
            elif hasattr(ds, 'InstanceNumber'):
                slice_location = ds.InstanceNumber
            else:
                slice_location = len(slices)  # 使用索引作为位置
            
            slices.append((slice_location, pixel_array))
        except Exception as e:
            print(f"Warning: Failed to read {dicom_file}: {e}")
            continue
    
    if len(slices) == 0:
        return None
    
    # 按切片位置排序
    slices.sort(key=lambda x: x[0])
    
    # 组合成3D体积
    volume = np.stack([s[1] for s in slices], axis=-1)  # (H, W, D)
    
    # 归一化到[0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    return volume


def load_png_mask(mask_dir, modality='MR'):
    """
    加载PNG标注文件，组合成3D标签
    
    Args:
        mask_dir: PNG标注文件目录
        modality: 'CT' 或 'MR'
    
    Returns:
        mask: (H, W, D) numpy数组，标签值
    """
    png_files = sorted(glob(os.path.join(mask_dir, '*.png')))
    
    if len(png_files) == 0:
        print(f"Warning: No PNG files found in {mask_dir}")
        return None
    
    masks = []
    for png_file in png_files:
        try:
            # 读取PNG文件
            img = Image.open(png_file)
            img_array = np.array(img)
            
            # 如果是RGB，转换为灰度
            if len(img_array.shape) == 3:
                # CHAOS MR数据的标注是灰度图，每个器官有不同灰度值
                # 转换为单通道
                img_array = img_array[:, :, 0] if img_array.shape[2] > 1 else img_array[:, :, 0]
            
            # 根据模态转换标签值
            if modality == 'MR':
                # MR数据：根据定义文件，标签值范围：
                # Liver: 63 (55-70)
                # Right kidney: 126 (110-135)
                # Left kidney: 189 (175-200)
                # Spleen: 252 (240-255)
                # 转换为类别标签：0=背景, 1=Liver, 2=Right Kidney, 3=Left Kidney, 4=Spleen
                label_mask = np.zeros_like(img_array, dtype=np.uint8)
                label_mask[(img_array >= 55) & (img_array <= 70)] = 1  # Liver
                label_mask[(img_array >= 110) & (img_array <= 135)] = 2  # Right Kidney
                label_mask[(img_array >= 175) & (img_array <= 200)] = 3  # Left Kidney
                label_mask[(img_array >= 240) & (img_array <= 255)] = 4  # Spleen
                img_array = label_mask
            elif modality == 'CT':
                # CT数据：只有liver标注，任何值>0都是liver
                # 转换为：0=背景, 1=Liver
                label_mask = np.zeros_like(img_array, dtype=np.uint8)
                label_mask[img_array > 0] = 1
                img_array = label_mask
            
            masks.append(img_array)
        except Exception as e:
            print(f"Warning: Failed to read {png_file}: {e}")
            continue
    
    if len(masks) == 0:
        return None
    
    # 组合成3D标签
    mask = np.stack(masks, axis=-1)  # (H, W, D)
    
    return mask.astype(np.uint8)


def process_chaos_ct_case(case_dir, case_name, output_dir):
    """
    处理CHAOS CT数据的一个病例
    
    Args:
        case_dir: 病例目录（如 CT/1/）
        case_name: 病例名称（如 case_1）
        output_dir: 输出目录
    """
    dicom_dir = os.path.join(case_dir, 'DICOM_anon')
    mask_dir = os.path.join(case_dir, 'Ground')
    
    # 检查目录是否存在
    if not os.path.exists(dicom_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Missing directories for {case_name}")
        return False
    
    # 加载DICOM图像
    volume = load_dicom_series(dicom_dir)
    if volume is None:
        print(f"Warning: Failed to load DICOM for {case_name}")
        return False
    
    # 加载PNG标注
    mask = load_png_mask(mask_dir, modality='CT')
    if mask is None:
        print(f"Warning: Failed to load mask for {case_name}")
        return False
    
    # 确保图像和标签尺寸匹配
    if volume.shape[:2] != mask.shape[:2]:
        # 调整标签尺寸
        H, W = volume.shape[:2]
        mask_resized = []
        for i in range(mask.shape[2]):
            mask_slice = cv2.resize(mask[:, :, i], (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resized.append(mask_slice)
        mask = np.stack(mask_resized, axis=-1)
    
    # 确保深度维度匹配
    min_depth = min(volume.shape[2], mask.shape[2])
    volume = volume[:, :, :min_depth]
    mask = mask[:, :, :min_depth]
    
    # 保存为H5文件（3D体积，用于测试）
    output_path = os.path.join(output_dir, 'test_vol_h5', f'{case_name}.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('image', data=volume.astype(np.float32))
        f.create_dataset('label', data=mask.astype(np.uint8))
    
    # 生成2D切片（用于训练）
    slices_dir = os.path.join(output_dir, 'train_slices')
    os.makedirs(slices_dir, exist_ok=True)
    
    slice_count = 0
    for i in range(volume.shape[2]):
        slice_image = volume[:, :, i]
        slice_mask = mask[:, :, i]
        
        # 跳过空白切片
        if slice_mask.sum() == 0:
            continue
        
        slice_name = f'{case_name}_slice{i:03d}'
        slice_path = os.path.join(slices_dir, f'{slice_name}.h5')
        
        with h5py.File(slice_path, 'w') as f:
            f.create_dataset('image', data=slice_image.astype(np.float32))
            f.create_dataset('label', data=slice_mask.astype(np.uint8))
        
        slice_count += 1
    
    print(f"✅ Processed {case_name}: {volume.shape[2]} slices, {slice_count} valid slices")
    return True


def process_chaos_mr_case(case_dir, case_name, output_dir, sequence='T1DUAL'):
    """
    处理CHAOS MR数据的一个病例
    
    Args:
        case_dir: 病例目录（如 MR/1/）
        case_name: 病例名称（如 case_1）
        output_dir: 输出目录
        sequence: 序列类型（'T1DUAL' 或 'T2SPIR'）
    """
    # MR数据的DICOM路径
    if sequence == 'T1DUAL':
        # T1DUAL有InPhase和OutPhase子目录，我们使用InPhase
        dicom_dir = os.path.join(case_dir, sequence, 'DICOM_anon', 'InPhase')
    else:
        # T2SPIR直接在DICOM_anon目录下
        dicom_dir = os.path.join(case_dir, sequence, 'DICOM_anon')
    
    mask_dir = os.path.join(case_dir, sequence, 'Ground')
    
    # 检查目录是否存在
    if not os.path.exists(dicom_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Missing directories for {case_name} {sequence}")
        return False
    
    # 加载DICOM图像
    # 注意：对于T1DUAL，dicom_dir已经指向InPhase目录，所以use_inphase参数不再需要
    volume = load_dicom_series(dicom_dir, use_inphase=True)
    if volume is None:
        print(f"Warning: Failed to load DICOM for {case_name} {sequence}")
        return False
    
    # 加载PNG标注
    mask = load_png_mask(mask_dir, modality='MR')
    if mask is None:
        print(f"Warning: Failed to load mask for {case_name} {sequence}")
        return False
    
    # 确保图像和标签尺寸匹配
    if volume.shape[:2] != mask.shape[:2]:
        H, W = volume.shape[:2]
        mask_resized = []
        for i in range(mask.shape[2]):
            mask_slice = cv2.resize(mask[:, :, i], (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resized.append(mask_slice)
        mask = np.stack(mask_resized, axis=-1)
    
    # 确保深度维度匹配
    min_depth = min(volume.shape[2], mask.shape[2])
    volume = volume[:, :, :min_depth]
    mask = mask[:, :, :min_depth]
    
    # 保存为H5文件（3D体积，用于测试）
    output_path = os.path.join(output_dir, 'test_vol_h5', f'{case_name}_{sequence}.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('image', data=volume.astype(np.float32))
        f.create_dataset('label', data=mask.astype(np.uint8))
    
    # 生成2D切片（用于训练）
    slices_dir = os.path.join(output_dir, 'train_slices')
    os.makedirs(slices_dir, exist_ok=True)
    
    slice_count = 0
    for i in range(volume.shape[2]):
        slice_image = volume[:, :, i]
        slice_mask = mask[:, :, i]
        
        # 跳过空白切片
        if slice_mask.sum() == 0:
            continue
        
        slice_name = f'{case_name}_{sequence}_slice{i:03d}'
        slice_path = os.path.join(slices_dir, f'{slice_name}.h5')
        
        with h5py.File(slice_path, 'w') as f:
            f.create_dataset('image', data=slice_image.astype(np.float32))
            f.create_dataset('label', data=slice_mask.astype(np.uint8))
        
        slice_count += 1
    
    print(f"✅ Processed {case_name} {sequence}: {volume.shape[2]} slices, {slice_count} valid slices")
    return True


def create_list_files(output_dir, train_slices_dir, test_vol_dir):
    """
    创建训练和测试列表文件
    """
    # 训练切片列表
    slice_files = sorted([f.replace('.h5', '') for f in os.listdir(train_slices_dir) if f.endswith('.h5')])
    train_list_path = os.path.join(output_dir, 'train_slices.list')
    with open(train_list_path, 'w') as f:
        for item in slice_files:
            f.write(f'{item}\n')
    print(f"✅ Created {train_list_path} with {len(slice_files)} slices")
    
    # 测试体积列表
    vol_files = sorted([f.replace('.h5', '') for f in os.listdir(test_vol_dir) if f.endswith('.h5')])
    test_list_path = os.path.join(output_dir, 'test.list')
    with open(test_list_path, 'w') as f:
        for item in vol_files:
            f.write(f'{item}\n')
    print(f"✅ Created {test_list_path} with {len(vol_files)} volumes")
    
    # 验证列表（从训练集中划分前20%作为验证集）
    val_size = max(1, len(vol_files) // 5)
    val_list_path = os.path.join(output_dir, 'val.list')
    with open(val_list_path, 'w') as f:
        for item in vol_files[:val_size]:
            f.write(f'{item}\n')
    print(f"✅ Created {val_list_path} with {val_size} volumes")


def main():
    parser = argparse.ArgumentParser(description='CHAOS数据集预处理')
    parser.add_argument('--data_dir', type=str, default='../data/CHAOS',
                        help='CHAOS数据集根目录')
    parser.add_argument('--output_dir', type=str, default='../data/CHAOS',
                        help='输出目录')
    parser.add_argument('--modality', type=str, choices=['CT', 'MR', 'both'], default='MR',
                        help='处理哪种模态（CT, MR, 或 both）')
    parser.add_argument('--mr_sequence', type=str, choices=['T1DUAL', 'T2SPIR', 'both'], default='T1DUAL',
                        help='MR序列类型（仅当modality=MR时有效）')
    
    args = parser.parse_args()
    
    # 规范化路径（处理Windows路径分隔符问题）
    # 使用os.path.abspath会自动解析相对路径，基于当前工作目录
    # 但为了更可靠，我们可以先切换到脚本所在目录的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # code/dataloaders -> code -> project root
    
    # 如果args.data_dir是相对路径，则相对于项目根目录
    if not os.path.isabs(args.data_dir):
        # 相对路径相对于当前工作目录解析
        data_dir = os.path.normpath(os.path.abspath(args.data_dir))
    else:
        data_dir = os.path.normpath(os.path.abspath(args.data_dir))
    
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.normpath(os.path.abspath(args.output_dir))
    else:
        output_dir = os.path.normpath(os.path.abspath(args.output_dir))
    
    # 构建训练集目录路径
    train_sets_dir = os.path.join(data_dir, 'CHAOS_Train_Sets', 'Train_Sets')
    train_sets_dir = os.path.normpath(train_sets_dir)
    
    # 检查数据目录是否存在
    if not os.path.exists(train_sets_dir):
        print(f"❌ 错误: 数据目录不存在: {train_sets_dir}")
        print(f"   请检查 --data_dir 参数是否正确")
        print(f"   当前数据目录: {data_dir}")
        print(f"   请确保数据目录结构为: {data_dir}/CHAOS_Train_Sets/Train_Sets/")
        return
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train_slices'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test_vol_h5'), exist_ok=True)
    
    print("=" * 60)
    print("CHAOS数据集预处理")
    print("=" * 60)
    print(f"数据目录: {train_sets_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模态: {args.modality}")
    print("=" * 60)
    
    processed_count = 0
    
    # 处理CT数据
    if args.modality in ['CT', 'both']:
        print("\n📂 处理CT数据...")
        ct_dir = os.path.join(train_sets_dir, 'CT')
        ct_dir = os.path.normpath(ct_dir)
        print(f"   检查CT目录: {ct_dir}")
        print(f"   目录存在: {os.path.exists(ct_dir)}")
        
        if os.path.exists(ct_dir):
            ct_cases = sorted([d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d))])
            for case_id in tqdm(ct_cases, desc='Processing CT cases'):
                case_dir = os.path.join(ct_dir, case_id)
                case_name = f'CT_{case_id}'
                if process_chaos_ct_case(case_dir, case_name, output_dir):
                    processed_count += 1
        else:
            print(f"Warning: CT directory not found: {ct_dir}")
    
    # 处理MR数据
    if args.modality in ['MR', 'both']:
        print("\n📂 处理MR数据...")
        mr_dir = os.path.join(train_sets_dir, 'MR')
        mr_dir = os.path.normpath(mr_dir)
        print(f"   检查MR目录: {mr_dir}")
        print(f"   目录存在: {os.path.exists(mr_dir)}")
        
        if os.path.exists(mr_dir):
            mr_cases = sorted([d for d in os.listdir(mr_dir) if os.path.isdir(os.path.join(mr_dir, d))])
            
            sequences = []
            if args.mr_sequence == 'both':
                sequences = ['T1DUAL', 'T2SPIR']
            else:
                sequences = [args.mr_sequence]
            
            for case_id in tqdm(mr_cases, desc='Processing MR cases'):
                case_dir = os.path.join(mr_dir, case_id)
                for sequence in sequences:
                    case_name = f'MR_{case_id}_{sequence}'
                    if process_chaos_mr_case(case_dir, case_name, output_dir, sequence):
                        processed_count += 1
        else:
            print(f"Warning: MR directory not found: {mr_dir}")
    
    # 创建列表文件
    print("\n📝 创建列表文件...")
    train_slices_dir = os.path.join(output_dir, 'train_slices')
    test_vol_dir = os.path.join(output_dir, 'test_vol_h5')
    create_list_files(output_dir, train_slices_dir, test_vol_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ 预处理完成！处理了 {processed_count} 个病例")
    print("=" * 60)
    print(f"训练切片目录: {train_slices_dir}")
    print(f"测试体积目录: {test_vol_dir}")
    print(f"列表文件: {output_dir}/train_slices.list, {output_dir}/test.list, {output_dir}/val.list")


if __name__ == '__main__':
    main()

