## Usage

### ACDC

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

## Custom dataset configuration

Copy the existing configuration file and modify the following parameters:

1. **base_config**:
   - `root_path`: Data set path
   - `exp`: Experiment Name
   - `num_classes`: Number of categories (including the background)
   - `labeled_num`: Indicate the quantity of data

2. **output_dir**: 实验结果保存目录

3. **hyperparameters**: 根据数据集特点调整超参数范围

