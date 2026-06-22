"""
超参数实验自动化运行脚本
支持网格搜索和随机搜索
"""
import os
import sys
import json
import subprocess
import time
import itertools
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from tqdm import tqdm

class HyperparameterExperiment:
    def __init__(self, config_path):
        """初始化实验配置"""
        # 处理相对路径
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            code_dir = os.path.dirname(script_dir)
            config_path = os.path.join(code_dir, config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.experiment_name = self.config['experiment_name']
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建实验日志目录
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建结果目录
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验记录文件
        self.experiment_log = self.output_dir / "experiment_log.json"
        
    def generate_hyperparameter_combinations(self):
        """生成所有超参数组合"""
        # 检查是否有自定义配置（3个配置模式）
        if 'custom_configs' in self.config:
            combinations = []
            config_names = self.config['hyperparameters'].get('config_name', {}).get('values', [])
            for config_name in config_names:
                if config_name in self.config['custom_configs']:
                    combo = self.config['custom_configs'][config_name].copy()
                    combo['_config_name'] = config_name  # 保存配置名称用于显示
                    combinations.append(combo)
            return combinations
        
        # 原有的网格搜索模式
        hyperparams = self.config['hyperparameters']
        combinations = []
        
        # 提取所有超参数的值
        param_names = []
        param_values = []
        
        for param_name, param_config in hyperparams.items():
            if param_config['type'] == 'grid':
                param_names.append(param_name)
                param_values.append(param_config['values'])
        
        # 生成所有组合
        for combo in itertools.product(*param_values):
            combo_dict = dict(zip(param_names, combo))
            combinations.append(combo_dict)
        
        return combinations
    
    def build_command(self, combo, exp_id):
        """构建训练命令"""
        base_config = self.config['base_config']
        cmd = [
            sys.executable, "train_VIM_deepsuperv.py",
            "--root_path", base_config['root_path'],
            "--exp", f"{base_config['exp']}_exp{exp_id}",
            "--model", base_config['model'],
            "--num_classes", str(base_config['num_classes']),
            "--max_iterations", str(base_config['max_iterations']),
            "--labeled_num", str(base_config['labeled_num']),
            "--seed", str(base_config['seed']),
            "--deterministic", str(base_config['deterministic']),
        ]
        
        # 添加超参数
        if 'batch_size' in combo:
            cmd.extend(["--batch_size", str(combo['batch_size'])])
        if 'base_lr' in combo:
            cmd.extend(["--base_lr", str(combo['base_lr'])])
        if 'patch_size' in combo:
            patch_size = combo['patch_size']
            cmd.extend(["--patch_size", str(patch_size[0]), str(patch_size[1])])
        
        # 通过--opts传递模型架构超参数
        opts = []
        if 'num_cls_tokens' in combo:
            opts.extend(["MODEL.NUM_CLS_TOKENS", str(combo['num_cls_tokens'])])
        if 'drop_path_rate' in combo:
            opts.extend(["MODEL.DROP_PATH_RATE", str(combo['drop_path_rate'])])
        
        if opts:
            cmd.extend(["--opts"] + opts)
        
        return cmd
    
    def save_experiment_info(self, exp_id, combo, status="running"):
        """保存实验信息"""
        if not self.experiment_log.exists():
            experiments = {}
        else:
            with open(self.experiment_log, 'r', encoding='utf-8') as f:
                experiments = json.load(f)
        
        experiments[f"exp_{exp_id}"] = {
            "id": exp_id,
            "hyperparameters": combo,
            "status": status,
            "start_time": datetime.now().isoformat(),
            "command": " ".join(self.build_command(combo, exp_id))
        }
        
        with open(self.experiment_log, 'w', encoding='utf-8') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
    
    def run_experiment(self, combo, exp_id, pbar=None):
        """运行单个实验"""
        # 提取配置名称（如果存在）
        config_name = combo.pop('_config_name', None)
        
        # 显示实验信息
        exp_info = f"实验 {exp_id}: "
        if config_name:
            exp_info += f"{config_name} "
        for key, value in combo.items():
            if not key.startswith('_') and key != 'description':  # 跳过内部标记和描述
                exp_info += f"{key}={value} "
        
        if pbar:
            pbar.set_description(exp_info[:60])  # 限制长度
        
        total_exps = pbar.total if pbar and hasattr(pbar, 'total') else '?'
        print(f"\n{'='*80}")
        print(f"[实验 {exp_id}/{total_exps}] 开始")
        if config_name:
            print(f"配置名称: {config_name}")
        print(f"{'='*80}")
        print(f"超参数配置:")
        for key, value in combo.items():
            if not key.startswith('_') and key != 'description':  # 跳过内部标记和描述
                print(f"  {key}: {value}")
        if 'description' in combo:
            print(f"  说明: {combo['description']}")
        print(f"{'='*80}")
        
        # 保存实验信息（移除内部标记）
        save_combo = {k: v for k, v in combo.items() if not k.startswith('_')}
        if config_name:
            save_combo['config_name'] = config_name
        self.save_experiment_info(exp_id, save_combo, "running")
        
        # 构建命令
        cmd = self.build_command(combo, exp_id)
        
        # 日志文件
        log_file = self.log_dir / f"exp_{exp_id}.log"
        
        # 运行训练
        start_time = time.time()
        try:
            # 获取code目录（脚本在code/hyperparameter_experiments/，需要回到code/）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            code_dir = os.path.dirname(script_dir)  # 回到code目录
            
            # 使用subprocess.Popen以便实时监控
            import io
            process = subprocess.Popen(
                cmd,
                cwd=code_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时写入日志并显示进度
            with open(log_file, 'w', encoding='utf-8', errors='ignore') as f:
                last_iteration = 0
                print(f"[实验 {exp_id}] 开始训练，正在初始化...")
                
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    
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
                                        
                                        # 提取loss信息
                                        loss_info = ""
                                        if 'Total_Loss=' in line:
                                            try:
                                                loss_part = line.split('Total_Loss=')[1].split()[0]
                                                loss_info = f"Loss={loss_part}"
                                            except:
                                                pass
                                        
                                        # 更新进度条
                                        if pbar:
                                            max_iter = self.config['base_config'].get('max_iterations', 10000)
                                            progress_pct = min(iter_num / max_iter * 100, 99)
                                            pbar.set_postfix({
                                                'iter': f"{iter_num}/{max_iter}",
                                                'progress': f"{progress_pct:.1f}%",
                                                'loss': loss_info[:15] if loss_info else ''
                                            })
                                        
                                        # 每1000个iteration打印一次关键信息
                                        if iter_num % 1000 == 0:
                                            print(f"\n[实验 {exp_id}] iteration {iter_num}/{max_iter} ({progress_pct:.1f}%) {loss_info}")
                                        
                        except (ValueError, IndexError) as e:
                            pass
                    
                    # 显示验证信息
                    if '[验证]' in line or '验证完成' in line:
                        print(f"\n[实验 {exp_id}] {line.strip()}")
                    
                    # 显示错误信息
                    if 'error' in line.lower() or 'Error' in line or 'Traceback' in line:
                        print(f"\n[实验 {exp_id}] [警告] {line.strip()}")
            
            # 等待进程完成
            process.wait()
            elapsed_time = time.time() - start_time
            
            if process.returncode == 0:
                status = "completed"
                elapsed_str = f"{elapsed_time/60:.1f}分钟" if elapsed_time < 3600 else f"{elapsed_time/3600:.2f}小时"
                print(f"[OK] 实验 {exp_id} 完成 (耗时: {elapsed_str})")
                if pbar:
                    pbar.set_postfix({'status': '完成', 'time': elapsed_str})
            else:
                status = "failed"
                print(f"[失败] 实验 {exp_id} 失败 (返回码: {process.returncode})")
                if pbar:
                    pbar.set_postfix({'status': '失败'})
            
            # 更新实验状态
            self.update_experiment_status(exp_id, status, elapsed_time)
            
        except Exception as e:
            print(f"[错误] 实验 {exp_id} 出错: {e}")
            if pbar:
                pbar.set_postfix({'status': '错误'})
            self.update_experiment_status(exp_id, "error", 0)
    
    def update_experiment_status(self, exp_id, status, elapsed_time):
        """更新实验状态"""
        if not self.experiment_log.exists():
            return
        
        with open(self.experiment_log, 'r', encoding='utf-8') as f:
            experiments = json.load(f)
        
        if f"exp_{exp_id}" in experiments:
            experiments[f"exp_{exp_id}"]["status"] = status
            experiments[f"exp_{exp_id}"]["elapsed_time"] = elapsed_time
            experiments[f"exp_{exp_id}"]["end_time"] = datetime.now().isoformat()
        
        with open(self.experiment_log, 'w', encoding='utf-8') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
    
    def run_all(self, start_from=0, max_experiments=None):
        """运行所有实验"""
        combinations = self.generate_hyperparameter_combinations()
        total = len(combinations)
        
        print(f"\n{'='*80}")
        print(f"[超参数实验计划]")
        print(f"{'='*80}")
        print(f"实验名称: {self.experiment_name}")
        print(f"总实验数: {total}")
        print(f"从实验 {start_from} 开始")
        if max_experiments:
            print(f"最多运行: {max_experiments} 个实验")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}\n")
        
        # 保存实验计划
        plan_file = self.output_dir / "experiment_plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "total_experiments": total,
                "combinations": combinations,
                "config": self.config
            }, f, indent=2, ensure_ascii=False)
        
        # 运行实验
        end_idx = total
        if max_experiments:
            end_idx = min(start_from + max_experiments, total)
        
        actual_total = end_idx - start_from
        experiments_to_run = combinations[start_from:end_idx]
        
        # 创建总体进度条
        with tqdm(total=actual_total, desc="总体进度", unit="实验", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            start_all_time = time.time()
            
            for idx, combo in enumerate(experiments_to_run, start=start_from):
                # 运行单个实验
                self.run_experiment(combo, idx, pbar)
                
                # 更新进度条
                pbar.update(1)
                
                # 计算预计剩余时间
                if idx > start_from:
                    elapsed_all = time.time() - start_all_time
                    avg_time_per_exp = elapsed_all / (idx - start_from + 1)
                    remaining_exps = actual_total - (idx - start_from + 1)
                    estimated_remaining = avg_time_per_exp * remaining_exps
                    
                    if estimated_remaining < 3600:
                        time_str = f"{estimated_remaining/60:.1f}分钟"
                    else:
                        time_str = f"{estimated_remaining/3600:.2f}小时"
                    
                    pbar.set_postfix({
                        '剩余时间': time_str,
                        '已完成': f"{idx - start_from + 1}/{actual_total}"
                    })
        
        total_time = time.time() - start_all_time
        total_time_str = f"{total_time/60:.1f}分钟" if total_time < 3600 else f"{total_time/3600:.2f}小时"
        
        print(f"\n{'='*80}")
        print(f"[OK] 所有实验完成！")
        print(f"总耗时: {total_time_str}")
        print(f"结果保存在: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='运行超参数实验')
    parser.add_argument('--config', type=str, 
                       default='hyperparameter_experiments/experiment_config.json',
                       help='实验配置文件路径')
    parser.add_argument('--start_from', type=int, default=0,
                       help='从第几个实验开始（用于断点续跑）')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='最多运行多少个实验')
    
    args = parser.parse_args()
    
    # 切换到code目录（脚本在code/hyperparameter_experiments/，需要回到code/）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    os.chdir(code_dir)
    
    # 运行实验
    experiment = HyperparameterExperiment(args.config)
    experiment.run_all(start_from=args.start_from, max_experiments=args.max_experiments)


if __name__ == "__main__":
    main()

