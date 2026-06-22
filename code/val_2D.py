import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    """
    计算单个类别的Dice和HD95指标
    处理ground truth为空的情况
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    # 情况1: 两者都为0（完美匹配）
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    
    # 情况2: 预测为0但gt不为0（完全漏检）
    if pred.sum() == 0 and gt.sum() > 0:
        return 0.0, 1000.0  # HD95设为很大的值
    
    # 情况3: 预测不为0但gt为0（假阳性）
    if pred.sum() > 0 and gt.sum() == 0:
        return 0.0, 1000.0  # HD95设为很大的值
    
    # 情况4: 两者都不为0（正常情况）
    try:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    except RuntimeError as e:
        # 如果medpy计算失败，返回默认值
        print(f"Warning: HD95 calculation failed: {e}, using default values")
        dice = metric.binary.dc(pred, gt) if pred.sum() > 0 and gt.sum() > 0 else 0.0
        return dice, 1000.0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    """
    测试单个体积（支持2D和3D）
    处理Deep Supervision的输出
    """
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # 如果是2D，添加一个维度
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]
    
    prediction = np.zeros_like(label)
    num_slices = image.shape[0]
    # 对于3D数据，逐切片处理（可能很慢，但这是必要的）
    for ind in range(num_slices):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output = net(input)
            # 处理Deep Supervision的输出（可能是tuple）
            if isinstance(output, tuple):
                output = output[0]  # 只使用主输出
            
            out = torch.argmax(torch.softmax(
                output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    """
    测试单个体积（Deep Supervision版本）
    处理Deep Supervision的输出
    """
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # 如果是2D，添加一个维度
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]
    
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output = net(input)
            # 处理Deep Supervision的输出
            if isinstance(output, tuple):
                output_main = output[0]  # 主输出
            else:
                output_main = output
            
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
