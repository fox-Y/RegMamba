@echo off
REM 超参数实验快速启动脚本 (Windows)

echo ========================================
echo 超参数实验快速启动
echo ========================================
echo.

cd /d %~dp0\..

echo 选择实验模式:
echo 1. 快速搜索 (18个实验，推荐)
echo 2. 完整搜索 (324个实验，耗时)
echo 3. 分析已有结果
echo 4. 从断点继续
echo.

set /p choice=请输入选项 (1-4): 

if "%choice%"=="1" (
    echo.
    echo 启动快速搜索...
    python code\hyperparameter_experiments\run_experiment.py --config code\hyperparameter_experiments\quick_search_config.json
) else if "%choice%"=="2" (
    echo.
    echo 启动完整搜索...
    python code\hyperparameter_experiments\run_experiment.py --config code\hyperparameter_experiments\experiment_config.json
) else if "%choice%"=="3" (
    echo.
    echo 分析实验结果...
    python code\hyperparameter_experiments\analyze_results.py --experiment_dir ..\experiments\hyperparameter_search --plot
) else if "%choice%"=="4" (
    set /p start_from=从第几个实验开始: 
    set /p max_exp=最多运行几个实验 (留空表示全部): 
    if "%max_exp%"=="" (
        python code\hyperparameter_experiments\run_experiment.py --config code\hyperparameter_experiments\experiment_config.json --start_from %start_from%
    ) else (
        python code\hyperparameter_experiments\run_experiment.py --config code\hyperparameter_experiments\experiment_config.json --start_from %start_from% --max_experiments %max_exp%
    )
) else (
    echo 无效选项
)

pause

