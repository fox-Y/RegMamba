"""
运行3个超参数配置的脚本
- config1: 从exp_0继续训练（5000->10000迭代）
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
        "--root_path", "../data/ACDC",
        "--exp", f"ACDC/VIM_140_labeled_{config_name}",
        "--model", "mambaunet",
        "--num_classes", "4",
        "--max_iterations", "10000",
        "--batch_size", str(batch_size),
        "--base_lr", str(base_lr),
        "--labeled_num", "140",
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
    """运行测试并提取结果（使用test_VIM_reg_recycle.py）"""
    print(f"\n[测试] {config_name} - 开始测试...")
    
    # 查找模型路径
    exp_name = f"ACDC/VIM_140_labeled_{config_name}"
    snapshot_path = f"../model/{exp_name}_140_labeled/mambaunet_register{num_cls_tokens}_v4_deepsup"
    
    # 检查路径是否存在
    if not os.path.exists(snapshot_path):
        print(f"[警告] 模型路径不存在: {snapshot_path}")
        return None
    
    # 构建测试命令
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)  # 回到code目录
    
    cmd = [
        sys.executable, "test_VIM_reg_recycle.py",
        "--root_path", "../data/ACDC",
        "--snapshot_path", snapshot_path,
        "--num_classes", "4",
        "--patch_size", "224", "224",
        "--opts", "MODEL.NUM_CLS_TOKENS", str(num_cls_tokens)
    ]
    
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
        
        for line in output.split('\n'):
            # 查找 "Mean Dice: 0.xxxx"
            if 'Mean Dice' in line or 'mean_dice' in line.lower():
                try:
                    dice_match = re.search(r'([\d.]+)', line.split(':')[-1])
                    if dice_match:
                        dice = float(dice_match.group(1))
                except:
                    pass
            
            # 查找 "Mean HD95: xxxx.xx"
            if 'Mean HD95' in line or 'mean_hd95' in line.lower():
                try:
                    hd95_match = re.search(r'([\d.]+)', line.split(':')[-1])
                    if hd95_match:
                        hd95 = float(hd95_match.group(1))
                except:
                    pass
        
        if dice is not None:
            print(f"[OK] {config_name} 测试完成 - Dice: {dice:.4f}, HD95: {hd95:.2f}" if hd95 else f"[OK] {config_name} 测试完成 - Dice: {dice:.4f}")
            return {"dice": dice, "hd95": hd95}
        else:
            print(f"[警告] {config_name} 测试完成但无法提取结果")
            # 尝试从checkpoint文件名提取dice
            import glob
            checkpoint_dir = snapshot_path
            checkpoints = glob.glob(f"{checkpoint_dir}/iter_*_dice_*.pth")
            if checkpoints:
                best_dice = 0.0
                for cp in checkpoints:
                    try:
                        dice_match = re.search(r'dice_([\d.]+)\.pth', cp)
                        if dice_match:
                            dice_val = float(dice_match.group(1))
                            if dice_val > best_dice:
                                best_dice = dice_val
                    except:
                        pass
                if best_dice > 0:
                    return {"dice": best_dice, "hd95": None}
            return None
            
    except Exception as e:
        print(f"[错误] {config_name} 测试失败: {e}")
        return None

def compare_results(results):
    """对比所有配置的结果"""
    print(f"\n{'='*80}")
    print(f"[测试结果对比]")
    print(f"{'='*80}\n")
    
    # 打印表格
    print(f"{'配置':<12} {'Batch':<8} {'LR':<8} {'Tokens':<8} {'Dice':<10} {'HD95':<10} {'状态':<10}")
    print(f"{'-'*80}")
    
    for config_name, result in results.items():
        config_info = {
            "config1": {"batch": 4, "lr": 0.01, "tokens": 4},
            "config2": {"batch": 8, "lr": 0.01, "tokens": 6},
            "config3": {"batch": 4, "lr": 0.02, "tokens": 8}
        }
        
        info = config_info.get(config_name, {})
        batch = info.get("batch", "?")
        lr = info.get("lr", "?")
        tokens = info.get("tokens", "?")
        
        if result:
            dice = f"{result.get('dice', 0):.4f}" if result.get('dice') else "N/A"
            hd95 = f"{result.get('hd95', 0):.2f}" if result.get('hd95') else "N/A"
            status = "完成"
        else:
            dice = "N/A"
            hd95 = "N/A"
            status = "失败"
        
        print(f"{config_name:<12} {batch:<8} {lr:<8} {tokens:<8} {dice:<10} {hd95:<10} {status:<10}")
    
    # 找出最佳配置
    valid_results = {k: v for k, v in results.items() if v and v.get('dice')}
    if valid_results:
        best_config = max(valid_results.items(), key=lambda x: x[1].get('dice', 0))
        print(f"\n[最佳配置] {best_config[0]} (Dice: {best_config[1].get('dice', 0):.4f})")
    
    print(f"{'='*80}\n")

def main():
    print(f"\n{'='*80}")
    print(f"[3个超参数配置实验]")
    print(f"{'='*80}")
    print(f"配置1: 从exp_0继续训练 (5000->10000迭代)")
    print(f"配置2: 从头训练 (10000迭代)")
    print(f"配置3: 从头训练 (10000迭代)")
    print(f"{'='*80}\n")
    
    # 自动查找config1的最新checkpoint
    def find_latest_checkpoint(config_name, num_cls_tokens):
        """查找指定配置的最新checkpoint"""
        import glob
        
        # 可能的checkpoint路径
        exp_name = f"ACDC/VIM_140_labeled_{config_name}"
        checkpoint_dir = f"../model/{exp_name}_140_labeled/mambaunet_register{num_cls_tokens}_v4_deepsup"
        
        # 转换为绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        code_dir = os.path.dirname(script_dir)
        checkpoint_dir_abs = os.path.join(code_dir, checkpoint_dir)
        
        if not os.path.exists(checkpoint_dir_abs):
            return None
        
        # 查找所有checkpoint文件
        checkpoints = glob.glob(os.path.join(checkpoint_dir_abs, "iter_*_dice_*.pth"))
        if not checkpoints:
            # 也尝试查找best_model
            best_model = os.path.join(checkpoint_dir_abs, f"mambaunet_register{num_cls_tokens}_v4_deepsup_best_model.pth")
            if os.path.exists(best_model):
                return best_model
            return None
        
        # 找到dice最高的checkpoint
        best_checkpoint = None
        best_dice = 0.0
        best_iter = 0
        
        for cp in checkpoints:
            try:
                # 从文件名提取: iter_5000_dice_0.8832.pth
                filename = os.path.basename(cp)
                iter_match = re.search(r'iter_(\d+)_dice_([\d.]+)\.pth', filename)
                if iter_match:
                    iter_num = int(iter_match.group(1))
                    dice_val = float(iter_match.group(2))
                    # 优先选择dice最高的，如果dice相同则选择iteration最高的
                    if dice_val > best_dice or (dice_val == best_dice and iter_num > best_iter):
                        best_dice = dice_val
                        best_iter = iter_num
                        best_checkpoint = cp
            except:
                pass
        
        return best_checkpoint
    
    # 查找config1的checkpoint（从exp0或config1自身）
    config1_checkpoint = None
    
    # 首先尝试从exp0查找（之前的实验）
    exp0_checkpoint = "../model/ACDC/VIM_140_labeled_exp0_140_labeled/mambaunet_register4_v4_deepsup/iter_5000_dice_0.8832.pth"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    exp0_checkpoint_abs = os.path.join(code_dir, exp0_checkpoint)
    
    if os.path.exists(exp0_checkpoint_abs):
        config1_checkpoint = exp0_checkpoint_abs
        print(f"[找到checkpoint] exp_0: {exp0_checkpoint}")
    else:
        # 尝试从config1自身查找
        config1_checkpoint = find_latest_checkpoint("config1", 4)
        if config1_checkpoint:
            print(f"[找到checkpoint] config1: {config1_checkpoint}")
        else:
            print(f"[提示] 未找到config1的checkpoint，将从头训练")
    
    # 定义3个配置
    configs = [
        {
            "name": "config1",
            "batch_size": 4,
            "base_lr": 0.01,
            "num_cls_tokens": 4,
            "description": "保守配置：小batch，标准学习率，少tokens",
            "resume": config1_checkpoint
        },
        {
            "name": "config2",
            "batch_size": 8,
            "base_lr": 0.01,
            "num_cls_tokens": 6,
            "description": "大batch配置：大batch，标准学习率，中等tokens",
            "resume": None
        },
        {
            "name": "config3",
            "batch_size": 4,
            "base_lr": 0.02,
            "num_cls_tokens": 8,
            "description": "高学习率配置：小batch，高学习率，多tokens",
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
    output_file = "hyperparameter_experiments/3configs_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_time": total_time_str,
            "configs": configs,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"[结果已保存] {output_file}")
    
    print(f"\n实验结果保存在:")
    for config in configs:
        exp_name = f"ACDC/VIM_140_labeled_{config['name']}"
        print(f"  {config['name']}: ../model/{exp_name}_140_labeled/")

if __name__ == "__main__":
    # 切换到code目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    os.chdir(code_dir)
    
    main()

