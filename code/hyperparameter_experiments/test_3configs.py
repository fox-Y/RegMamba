"""
测试3个超参数配置的模型并对比结果
使用test_VIM_reg_recycle.py对每个配置进行测试
"""
import os
import sys
import subprocess
import re
from tqdm import tqdm

def run_test(config_name, num_cls_tokens):
    """运行单个配置的测试"""
    print(f"\n[测试] {config_name} - 开始测试...")
    print(f"    使用 {num_cls_tokens} 个Register Tokens")
    
    # 查找模型路径
    exp_name = f"ACDC/VIM_140_labeled_{config_name}"
    snapshot_path = f"../model/{exp_name}_140_labeled/mambaunet_register{num_cls_tokens}_v4_deepsup"
    
    # 转换为绝对路径检查
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    snapshot_path_abs = os.path.join(code_dir, snapshot_path)
    
    # 检查路径是否存在
    if not os.path.exists(snapshot_path_abs):
        print(f"[错误] 模型路径不存在: {snapshot_path}")
        return None
    
    # 构建测试命令（确保NUM_CLS_TOKENS正确传递）
    cmd = [
        sys.executable, "test_VIM_reg_recycle.py",
        "--root_path", "../data/ACDC",
        "--snapshot_path", snapshot_path,
        "--num_classes", "4",
        "--patch_size", "224", "224",
        "--opts", "MODEL.NUM_CLS_TOKENS", str(num_cls_tokens)
    ]
    
    print(f"    模型路径: {snapshot_path}")
    print(f"    命令: {' '.join(cmd)}")
    
    try:
        # 运行测试
        result = subprocess.run(
            cmd,
            cwd=code_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        # 从输出中提取结果
        output = result.stdout
        
        # 查找Mean Dice和Mean HD95
        dice = None
        hd95 = None
        per_class_results = []
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # 查找 "Mean Dice: 0.xxxx" 或 "mean_dice : 0.xxxx"
            if 'Mean Dice' in line or 'mean_dice' in line.lower():
                try:
                    # 尝试多种格式
                    dice_match = re.search(r'([\d.]+)', line.split(':')[-1])
                    if dice_match:
                        dice = float(dice_match.group(1))
                except:
                    pass
            
            # 查找 "Mean HD95: xxxx.xx" 或 "mean_hd95 : xxxx.xx"
            if 'Mean HD95' in line or 'mean_hd95' in line.lower():
                try:
                    hd95_match = re.search(r'([\d.]+)', line.split(':')[-1])
                    if hd95_match:
                        hd95 = float(hd95_match.group(1))
                except:
                    pass
            
            # 提取每个类别的结果
            if 'Class' in line and ('Dice' in line or 'dice' in line.lower()):
                try:
                    # 格式: "Class 1: Dice = 0.xxxx, HD95 = xxxx.xx"
                    class_match = re.search(r'Class\s+(\d+)', line)
                    if class_match:
                        class_num = int(class_match.group(1))
                        dice_match = re.search(r'Dice\s*=\s*([\d.]+)', line, re.IGNORECASE)
                        hd95_match = re.search(r'HD95\s*=\s*([\d.]+)', line, re.IGNORECASE)
                        if dice_match:
                            class_dice = float(dice_match.group(1))
                            class_hd95 = float(hd95_match.group(1)) if hd95_match else None
                            per_class_results.append({
                                'class': class_num,
                                'dice': class_dice,
                                'hd95': class_hd95
                            })
                except:
                    pass
        
        if dice is not None:
            print(f"[OK] {config_name} 测试完成")
            print(f"     Dice: {dice:.4f}, HD95: {hd95:.2f}" if hd95 else f"     Dice: {dice:.4f}")
            return {
                "dice": dice,
                "hd95": hd95,
                "per_class": per_class_results,
                "status": "完成"
            }
        else:
            print(f"[警告] {config_name} 测试完成但无法提取结果")
            return {
                "dice": None,
                "hd95": None,
                "per_class": [],
                "status": "失败"
            }
            
    except Exception as e:
        print(f"[错误] {config_name} 测试失败: {e}")
        return {
            "dice": None,
            "hd95": None,
            "per_class": [],
            "status": "错误"
        }

def compare_results(results):
    """对比所有配置的结果"""
    print(f"\n{'='*80}")
    print(f"[测试结果对比]")
    print(f"{'='*80}\n")
    
    # 配置信息
    config_info = {
        "config1": {"batch": 4, "lr": 0.01, "tokens": 4, "desc": "保守配置"},
        "config2": {"batch": 8, "lr": 0.01, "tokens": 6, "desc": "大batch配置"},
        "config3": {"batch": 4, "lr": 0.02, "tokens": 8, "desc": "高学习率配置"}
    }
    
    # 打印总体对比表格
    print(f"{'配置':<12} {'Batch':<8} {'LR':<8} {'Tokens':<8} {'Dice':<10} {'HD95':<10} {'状态':<10}")
    print(f"{'-'*80}")
    
    for config_name, result in results.items():
        info = config_info.get(config_name, {})
        batch = info.get("batch", "?")
        lr = info.get("lr", "?")
        tokens = info.get("tokens", "?")
        
        if result and result.get('dice') is not None:
            dice = f"{result['dice']:.4f}"
            hd95 = f"{result['hd95']:.2f}" if result.get('hd95') else "N/A"
            status = result.get('status', '完成')
        else:
            dice = "N/A"
            hd95 = "N/A"
            status = result.get('status', '失败') if result else '失败'
        
        print(f"{config_name:<12} {batch:<8} {lr:<8} {tokens:<8} {dice:<10} {hd95:<10} {status:<10}")
    
    # 找出最佳配置
    valid_results = {k: v for k, v in results.items() if v and v.get('dice') is not None}
    if valid_results:
        best_config = max(valid_results.items(), key=lambda x: x[1].get('dice', 0))
        print(f"\n[最佳配置] {best_config[0]} (Dice: {best_config[1].get('dice', 0):.4f})")
        if best_config[1].get('hd95'):
            print(f"            HD95: {best_config[1].get('hd95', 0):.2f}")
    
    # 打印每个类别的详细对比
    print(f"\n{'='*80}")
    print(f"[每个类别的详细对比]")
    print(f"{'='*80}\n")
    
    # 收集所有类别的结果
    all_classes = set()
    for result in results.values():
        if result and result.get('per_class'):
            for pc in result['per_class']:
                all_classes.add(pc['class'])
    
    if all_classes:
        all_classes = sorted(all_classes)
        print(f"{'类别':<8} {'Config1 Dice':<15} {'Config2 Dice':<15} {'Config3 Dice':<15} {'最佳':<10}")
        print(f"{'-'*80}")
        
        for class_num in all_classes:
            config1_dice = "N/A"
            config2_dice = "N/A"
            config3_dice = "N/A"
            best_config = None
            best_dice = 0.0
            
            for config_name in ['config1', 'config2', 'config3']:
                result = results.get(config_name)
                if result and result.get('per_class'):
                    for pc in result['per_class']:
                        if pc['class'] == class_num:
                            dice_val = pc['dice']
                            if config_name == 'config1':
                                config1_dice = f"{dice_val:.4f}"
                            elif config_name == 'config2':
                                config2_dice = f"{dice_val:.4f}"
                            elif config_name == 'config3':
                                config3_dice = f"{dice_val:.4f}"
                            
                            if dice_val > best_dice:
                                best_dice = dice_val
                                best_config = config_name
            
            best_mark = f"{best_config}" if best_config else "N/A"
            print(f"Class {class_num:<4} {config1_dice:<15} {config2_dice:<15} {config3_dice:<15} {best_mark:<10}")
    
    print(f"{'='*80}\n")

def main():
    print(f"\n{'='*80}")
    print(f"[3个超参数配置测试对比]")
    print(f"{'='*80}")
    print(f"将依次测试config1、config2、config3的训练结果")
    print(f"{'='*80}\n")
    
    # 定义3个配置
    configs = [
        {
            "name": "config1",
            "num_cls_tokens": 4,
            "description": "保守配置：batch=4, lr=0.01, tokens=4"
        },
        {
            "name": "config2",
            "num_cls_tokens": 6,
            "description": "大batch配置：batch=8, lr=0.01, tokens=6"
        },
        {
            "name": "config3",
            "num_cls_tokens": 8,
            "description": "高学习率配置：batch=4, lr=0.02, tokens=8"
        }
    ]
    
    # 运行测试
    results = {}
    with tqdm(total=len(configs), desc="测试进度", unit="配置") as pbar:
        for config in configs:
            result = run_test(config["name"], config["num_cls_tokens"])
            results[config["name"]] = result
            pbar.update(1)
    
    # 对比结果
    compare_results(results)
    
    # 保存结果到文件
    import json
    output_file = "hyperparameter_experiments/3configs_test_results.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "3configs_test_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "configs": configs,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[结果已保存] {output_path}")
    
    # 生成文本报告
    report_file = os.path.join(script_dir, "3configs_test_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("3个超参数配置测试结果报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("总体对比:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'配置':<12} {'Batch':<8} {'LR':<8} {'Tokens':<8} {'Dice':<10} {'HD95':<10} {'状态':<10}\n")
        f.write("-" * 80 + "\n")
        
        config_info = {
            "config1": {"batch": 4, "lr": 0.01, "tokens": 4},
            "config2": {"batch": 8, "lr": 0.01, "tokens": 6},
            "config3": {"batch": 4, "lr": 0.02, "tokens": 8}
        }
        
        for config_name, result in results.items():
            info = config_info.get(config_name, {})
            batch = info.get("batch", "?")
            lr = info.get("lr", "?")
            tokens = info.get("tokens", "?")
            
            if result and result.get('dice') is not None:
                dice = f"{result['dice']:.4f}"
                hd95 = f"{result['hd95']:.2f}" if result.get('hd95') else "N/A"
                status = result.get('status', '完成')
            else:
                dice = "N/A"
                hd95 = "N/A"
                status = result.get('status', '失败') if result else '失败'
            
            f.write(f"{config_name:<12} {batch:<8} {lr:<8} {tokens:<8} {dice:<10} {hd95:<10} {status:<10}\n")
        
        # 最佳配置
        valid_results = {k: v for k, v in results.items() if v and v.get('dice') is not None}
        if valid_results:
            best_config = max(valid_results.items(), key=lambda x: x[1].get('dice', 0))
            f.write(f"\n最佳配置: {best_config[0]} (Dice: {best_config[1].get('dice', 0):.4f})\n")
            if best_config[1].get('hd95'):
                f.write(f"         HD95: {best_config[1].get('hd95', 0):.2f}\n")
        
        # 每个类别的详细结果
        f.write("\n" + "=" * 80 + "\n")
        f.write("每个类别的详细结果:\n")
        f.write("=" * 80 + "\n\n")
        
        for config_name, result in results.items():
            if result and result.get('per_class'):
                f.write(f"{config_name}:\n")
                for pc in result['per_class']:
                    f.write(f"  Class {pc['class']}: Dice = {pc['dice']:.4f}")
                    if pc.get('hd95'):
                        f.write(f", HD95 = {pc['hd95']:.2f}")
                    f.write("\n")
                f.write("\n")
    
    print(f"[报告已保存] {report_file}")

if __name__ == "__main__":
    # 切换到code目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    os.chdir(code_dir)
    
    main()

