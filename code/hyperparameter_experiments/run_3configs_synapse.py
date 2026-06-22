"""
运行3个超参数配置的脚本（Synapse数据集）
- config1: 从头训练（10000迭代）
- config2: 从头训练（10000迭代）
- config3: 从头训练（10000迭代）

功能：
1. 每1000次迭代显示进度
2. 训练完成后自动测试并对比结果
"""
import os
import sys
import subprocess
import time
import json
import re
from pathlib import Path
from tqdm import tqdm

def run_experiment(config_name, batch_size, base_lr, num_cls_tokens, exp_id, resume_path=None, pbar=None):
    """运行单个实验，带实时进度显示"""
    print(f"\n{'='*80}")
    print(f"[实验 {exp_id}/3] {config_name}")
    print(f"{'='*80}")
    print(f"超参数配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  base_lr: {base_lr}")
    print(f"  num_cls_tokens: {num_cls_tokens}")
    if resume_path:
        print(f"  继续训练: 从 {resume_path} 继续")
    else:
        print(f"  从头训练: 10000次迭代")
    print(f"{'='*80}\n")
    
    # 构建命令
    cmd = [
        sys.executable, "train_VIM_deepsuperv.py",
        "--root_path", "../data/Synapse",
        "--exp", f"Synapse/VIM_18_labeled_{config_name}",
        "--model", "mambaunet",
        "--num_classes", "9",
        "--max_iterations", "10000",
        "--batch_size", str(batch_size),
        "--base_lr", str(base_lr),
        "--labeled_num", "18",
        "--seed", "1337",
        "--deterministic", "1",
        "--opts", "MODEL.NUM_CLS_TOKENS", str(num_cls_tokens)
    ]
    
    # 如果指定了resume路径，添加resume参数
    if resume_path:
        # 转换为相对于code目录的路径
        if os.path.isabs(resume_path):
            # 如果是绝对路径，转换为相对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            code_dir = os.path.dirname(script_dir)
            try:
                resume_path_rel = os.path.relpath(resume_path, code_dir)
            except ValueError:
                # 如果路径不在同一驱动器，使用绝对路径
                resume_path_rel = resume_path
        else:
            resume_path_rel = resume_path
        
        if os.path.exists(resume_path):
            cmd.extend(["--resume", resume_path_rel])
            print(f"[继续训练] 从checkpoint继续: {resume_path_rel}")
        else:
            print(f"[警告] checkpoint不存在: {resume_path}，将从头训练")
    
    # 运行训练
    start_time = time.time()
    try:
        # 获取code目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        code_dir = os.path.dirname(script_dir)  # 回到code目录
        
        # 创建实验进度条（简化显示）
        max_iter = 10000
        exp_pbar = tqdm(total=max_iter, desc=f"  {config_name}", unit="iter", leave=False, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        last_iteration = 0
        
        # 运行训练（实时捕获输出）
        process = subprocess.Popen(
            cmd,
            cwd=code_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取输出（静默模式，只提取关键信息）
        for line in process.stdout:
            # 尝试提取iteration信息
            if 'iteration' in line.lower():
                try:
                    # 解析格式: "iteration 123 : Total_Loss=..."
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'iteration' and i + 1 < len(parts):
                            iter_str = parts[i + 1].rstrip(':')
                            iter_num = int(iter_str)
                            if iter_num > last_iteration:
                                last_iteration = iter_num
                                
                                # 提取loss和dice信息
                                loss_info = ""
                                dice_info = ""
                                if 'Total_Loss=' in line:
                                    try:
                                        loss_part = line.split('Total_Loss=')[1].split()[0]
                                        loss_info = loss_part
                                    except:
                                        pass
                                
                                # 更新进度条
                                progress_pct = min(iter_num / max_iter * 100, 99)
                                exp_pbar.update(iter_num - exp_pbar.n)
                                
                                # 每1000个iteration更新一次显示
                                if iter_num % 1000 == 0:
                                    postfix_info = {
                                        '进度': f"{progress_pct:.1f}%",
                                        'loss': loss_info[:8] if loss_info else ''
                                    }
                                    exp_pbar.set_postfix(postfix_info)
                                
                except (ValueError, IndexError):
                    pass
            
            # 提取验证结果（Dice和HD95）
            if 'mean_dice' in line.lower() or ('dice' in line.lower() and 'hd95' in line.lower()):
                try:
                    # 尝试提取dice和hd95数值
                    dice_match = re.search(r'dice[:\s]+([\d.]+)', line, re.IGNORECASE)
                    hd95_match = re.search(r'hd95[:\s]+([\d.]+)', line, re.IGNORECASE)
                    if dice_match:
                        dice_val = float(dice_match.group(1))
                        hd95_val = float(hd95_match.group(1)) if hd95_match else None
                        # 更新进度条显示验证指标
                        postfix_info = {
                            '进度': f"{min(last_iteration / max_iter * 100, 99):.1f}%",
                            'Dice': f"{dice_val:.4f}"
                        }
                        if hd95_val:
                            postfix_info['HD95'] = f"{hd95_val:.2f}"
                        exp_pbar.set_postfix(postfix_info)
                except:
                    pass
        
        # 等待进程完成
        process.wait()
        exp_pbar.close()
        
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            elapsed_str = f"{elapsed_time/60:.1f}分钟" if elapsed_time < 3600 else f"{elapsed_time/3600:.2f}小时"
            print(f"\n[OK] {config_name} 训练完成 (耗时: {elapsed_str})")
            return True, elapsed_time
        else:
            print(f"\n[失败] {config_name} 训练失败 (返回码: {process.returncode})")
            return False, elapsed_time
            
    except Exception as e:
        print(f"\n[错误] {config_name} 出错: {e}")
        return False, 0

def run_test(config_name, num_cls_tokens):
    """运行测试并提取结果（使用test_synapse.py）"""
    print(f"\n[测试] {config_name} - 开始测试...")
    print(f"    使用 {num_cls_tokens} 个Register Tokens")
    
    # 查找模型路径
    exp_name = f"Synapse/VIM_18_labeled_{config_name}"
    model_path = f"../model/{exp_name}_18_labeled/mambaunet_register{num_cls_tokens}_v4_deepsup/mambaunet_register{num_cls_tokens}_v4_deepsup_best_model.pth"
    
    # 转换为绝对路径检查
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    model_path_abs = os.path.join(code_dir, model_path)
    
    # 检查路径是否存在
    if not os.path.exists(model_path_abs):
        print(f"[错误] 模型路径不存在: {model_path}")
        return None
    
    # 构建测试命令
    cmd = [
        sys.executable, "test_synapse.py",
        "--root_path", "../data/Synapse",
        "--exp", exp_name,
        "--model", "mambaunet",
        "--num_classes", "9",
        "--model_path", model_path,
        "--patch_size", "224", "224",
        "--opts", "MODEL.NUM_CLS_TOKENS", str(num_cls_tokens)
    ]
    
    print(f"    模型路径: {model_path}")
    
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
    
    # 配置信息（针对Synapse多类别数据集，batch size已优化）
    config_info = {
        "config1": {"batch": 6, "lr": 0.01, "tokens": 8, "desc": "保守多类别配置"},
        "config2": {"batch": 10, "lr": 0.01, "tokens": 10, "desc": "平衡多类别配置"},
        "config3": {"batch": 6, "lr": 0.015, "tokens": 12, "desc": "激进多类别配置"}
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
    
    print(f"{'='*80}\n")

def main():
    print(f"\n{'='*80}")
    print(f"[3个超参数配置实验 - Synapse数据集（多类别优化，Batch Size已优化）]")
    print(f"{'='*80}")
    print(f"配置1: 保守多类别 (Batch=6, LR=0.01, Tokens=8) - 稳定且更快")
    print(f"配置2: 平衡多类别 (Batch=10, LR=0.01, Tokens=10) [推荐] - 高效")
    print(f"配置3: 激进多类别 (Batch=6, LR=0.015, Tokens=12) - 高性能且稳定")
    print(f"所有配置: 从头训练 (10000迭代)")
    print(f"预计总时间: 约2-2.5小时（相比原配置节省约1-1.5小时）")
    print(f"{'='*80}\n")
    
    # 定义3个配置（针对Synapse多类别数据集优化，batch size已优化）
    configs = [
        {
            "name": "config1",
            "batch_size": 6,
            "base_lr": 0.01,
            "num_cls_tokens": 8,
            "description": "保守多类别配置：中等batch，标准学习率，8 tokens（稳定且更快）",
            "resume": None
        },
        {
            "name": "config2",
            "batch_size": 10,
            "base_lr": 0.01,
            "num_cls_tokens": 10,
            "description": "平衡多类别配置：大batch，标准学习率，10 tokens（高效推荐）",
            "resume": None
        },
        {
            "name": "config3",
            "batch_size": 6,
            "base_lr": 0.015,
            "num_cls_tokens": 12,
            "description": "激进多类别配置：中等batch，略高学习率，12 tokens（高性能且稳定）",
            "resume": None
        }
    ]
    
    # 创建进度条
    with tqdm(total=len(configs), desc="总体进度", unit="实验") as pbar:
        start_all_time = time.time()
        
        for idx, config in enumerate(configs, start=1):
            # 运行实验
            success, elapsed_time = run_experiment(
                config["name"],
                config["batch_size"],
                config["base_lr"],
                config["num_cls_tokens"],
                idx,
                config["resume"]
            )
            
            # 更新进度条
            pbar.update(1)
            
            # 计算预计剩余时间
            if idx > 1:
                elapsed_all = time.time() - start_all_time
                avg_time_per_exp = elapsed_all / idx
                remaining_exps = len(configs) - idx
                estimated_remaining = avg_time_per_exp * remaining_exps
                
                if estimated_remaining < 3600:
                    time_str = f"{estimated_remaining/60:.1f}分钟"
                else:
                    time_str = f"{estimated_remaining/3600:.2f}小时"
                
                pbar.set_postfix({
                    '剩余时间': time_str,
                    '已完成': f"{idx}/{len(configs)}"
                })
    
    total_time = time.time() - start_all_time
    total_time_str = f"{total_time/60:.1f}分钟" if total_time < 3600 else f"{total_time/3600:.2f}小时"
    
    print(f"\n{'='*80}")
    print(f"[OK] 所有训练完成！")
    print(f"总耗时: {total_time_str}")
    print(f"{'='*80}\n")
    
    # 运行测试并提取结果
    print(f"[开始测试所有配置...]")
    results = {}
    for config in configs:
        result = run_test(config["name"], config["num_cls_tokens"])
        results[config["name"]] = result
    
    # 对比结果
    compare_results(results)
    
    # 保存结果到JSON
    output_file = "hyperparameter_experiments/3configs_synapse_results.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "3configs_synapse_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_time": total_time_str,
            "configs": configs,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"[结果已保存] {output_path}")
    
    print(f"\n实验结果保存在:")
    for config in configs:
        exp_name = f"Synapse/VIM_18_labeled_{config['name']}"
        print(f"  {config['name']}: ../model/{exp_name}_18_labeled/")

if __name__ == "__main__":
    # 切换到code目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    os.chdir(code_dir)
    
    main()

