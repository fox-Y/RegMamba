# 数据集配置文件

这里存放不同数据集的超参数实验配置。

## 可用配置

### CHAOS (当前默认)
- **配置文件**: `../experiment_config.json` 或 `../quick_search_config.json`
- **类别数**: 5 (背景 + 4个器官)
- **标注数量**: 20个病例

### ACDC
- **配置文件**: `ACDC_config.json`
- **类别数**: 4 (背景 + 3个心脏结构)
- **标注数量**: 140个病例

### Synapse
- **配置文件**: `Synapse_config.json`
- **类别数**: 9 (背景 + 8个器官)
- **标注数量**: 18个病例

## 使用方法

### 在ACDC数据集上运行

```bash
cd code
python hyperparameter_experiments/run_experiment.py \
    --config hyperparameter_experiments/dataset_configs/ACDC_config.json
```

### 在Synapse数据集上运行

```bash
cd code
python hyperparameter_experiments/run_experiment.py \
    --config hyperparameter_experiments/dataset_configs/Synapse_config.json
```

### 在CHAOS数据集上运行（默认）

```bash
cd code
python hyperparameter_experiments/run_experiment.py \
    --config hyperparameter_experiments/quick_search_config.json
```

## 自定义数据集配置

复制现有配置文件，修改以下参数：

1. **base_config**:
   - `root_path`: 数据集路径
   - `exp`: 实验名称
   - `num_classes`: 类别数（包括背景）
   - `labeled_num`: 标注数据数量

2. **output_dir**: 实验结果保存目录

3. **hyperparameters**: 根据数据集特点调整超参数范围

