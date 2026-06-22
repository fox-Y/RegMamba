# 超参数实验框架

## 📁 文件结构

```
hyperparameter_experiments/
├── __init__.py                    # 包初始化文件
├── experiment_config.json         # 完整实验配置（324个实验）
├── quick_search_config.json      # 快速搜索配置（18个实验）
├── run_experiment.py              # 实验运行脚本
├── analyze_results.py             # 结果分析脚本
├── quick_start.bat                # Windows快速启动脚本
└── README.md                      # 本文件
```

## 🚀 快速开始

### 1. 快速搜索（推荐）

```bash
cd code
python hyperparameter_experiments/run_experiment.py \
    --config hyperparameter_experiments/quick_search_config.json
```

### 2. 完整搜索

```bash
cd code
python hyperparameter_experiments/run_experiment.py \
    --config hyperparameter_experiments/experiment_config.json
```

### 3. 分析结果

```bash
cd code
python hyperparameter_experiments/analyze_results.py \
    --experiment_dir ../experiments/hyperparameter_search \
    --plot
```

## 📊 实验配置说明

### 快速搜索配置 (quick_search_config.json)

- **实验数量**: 18个 (3×2×3)
- **超参数**: batch_size, base_lr, num_cls_tokens
- **训练迭代**: 5000 iterations
- **适用场景**: 快速找到大致最优范围

### 完整搜索配置 (experiment_config.json)

- **实验数量**: 324个 (3×3×3×4×3)
- **超参数**: batch_size, base_lr, patch_size, num_cls_tokens, drop_path_rate
- **训练迭代**: 10000 iterations
- **适用场景**: 全面搜索最佳配置

## 🔧 自定义配置

编辑JSON配置文件，修改超参数范围：

```json
{
  "hyperparameters": {
    "batch_size": {
      "type": "grid",
      "values": [4, 6, 8]  // 修改这里
    }
  }
}
```

## 📈 结果输出

实验结果保存在 `../experiments/hyperparameter_search/`:

- `logs/`: 每个实验的训练日志
- `results/`: 分析结果
  - `all_results.csv`: 所有实验结果表格
  - `all_results.xlsx`: Excel格式
  - `hyperparameter_analysis.txt`: 超参数影响分析
  - `experiment_summary.txt`: 实验总结
  - `plots/`: 可视化图表

## ⚠️ 注意事项

1. **显存管理**: 根据GPU显存调整batch_size和patch_size
2. **时间管理**: 完整搜索可能需要数天，建议先用快速搜索
3. **断点续跑**: 使用 `--start_from` 参数从指定位置继续
4. **结果备份**: 定期备份实验结果目录

## 📚 详细文档

查看 `📊 超参数实验流程指南.md` 获取完整的使用说明。

