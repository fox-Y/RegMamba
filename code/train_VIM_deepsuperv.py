import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
# from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_mamba import MambaUnet as VIM_seg

from config import get_config

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets_Synapse, BaseDataSets_CHAOS
# from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224],
                    help='patch size of network input (height width)')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
args = parser.parse_args()


config = get_config(args)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate" in dataset:
        # Prostate: 35个训练病例，940个训练切片，平均每个病例~27切片
        ref_dict = {"7": 189, "14": 378, "21": 567, "28": 756, "35": 940}
    elif "Synapse" in dataset:
        # Synapse: 18个训练病例，2211个训练切片，平均每个病例~123切片
        # 可以根据需要设置不同的标注数量
        ref_dict = {
            "3": 369, "6": 738, "9": 1107, 
            "12": 1476, "15": 1845, "18": 2211
        }
    elif "CHAOS" in dataset:
        # CHAOS: 20个训练病例，500个训练切片，平均每个病例~25切片
        ref_dict = {
            "5": 125, "10": 250, "15": 375, "20": 500
        }
    else:
        print(f"Error: Unsupported dataset {dataset}")
        return None
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)




    model = VIM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model.load_from(config)
    
    # 初始化optimizer（resume时需要）
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    # === Resume from checkpoint ===
    iter_num = 0
    best_performance = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 从checkpoint恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume)
            # 加载模型权重（兼容不同的保存格式）
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # 加载optimizer状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> 已恢复optimizer状态")
            
            # 恢复训练进度
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num']
                print(f"=> 从iteration {iter_num} 继续训练")
            elif 'iteration' in checkpoint:
                iter_num = checkpoint['iteration']
                print(f"=> 从iteration {iter_num} 继续训练")
            
            # 恢复最佳性能
            if 'best_performance' in checkpoint:
                best_performance = checkpoint['best_performance']
                print(f"=> 最佳性能: {best_performance:.4f}")
        else:
            print(f"=> 警告: checkpoint文件不存在: {args.resume}")
            print("=> 从头开始训练")




    # 根据数据集类型选择不同的DataLoader
    if "Synapse" in args.root_path:
        # Synapse数据集使用BaseDataSets_Synapse
        db_train = BaseDataSets_Synapse(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
        db_val = BaseDataSets_Synapse(base_dir=args.root_path, split="val")
    elif "CHAOS" in args.root_path:
        # CHAOS数据集使用BaseDataSets_CHAOS
        db_train = BaseDataSets_CHAOS(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
        db_val = BaseDataSets_CHAOS(base_dir=args.root_path, split="val")
    else:
        # ACDC和Prostate使用BaseDataSets
        db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
        db_val = BaseDataSets(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)  # num_workers = 0
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    # optimizer已在resume部分初始化
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    # === Deep Supervision 信息 ===
    if hasattr(model, 'use_deep_supervision') and model.use_deep_supervision:
        logging.info("=" * 60)
        logging.info("🔥 Deep Supervision ENABLED")
        logging.info(f"  - Auxiliary heads: {len(model.aux_heads)} layers")
        logging.info("  - Loss formula: Total = Main_Loss + 0.4 * Aux_Loss")
        logging.info("  - Main_Loss = 0.5 * (CE + Dice)")
        logging.info("  - Aux_Loss = Average of all auxiliary layer losses")
        logging.info("=" * 60)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # iter_num 和 best_performance 在resume时可能已经被设置
    if not args.resume:
        iter_num = 0
        best_performance = 0.0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # === Deep Supervision: 处理主输出和辅助输出 ===
            model_output = model(volume_batch)
            
            # 训练时返回 (main_output, aux_outputs)，测试时只返回 main_output
            if isinstance(model_output, tuple):
                outputs, aux_outputs = model_output
            else:
                outputs = model_output
                aux_outputs = None
            
            outputs_soft = torch.softmax(outputs, dim=1)
            
            # 主损失
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss_main = 0.5 * (loss_dice + loss_ce)
            
            # === Deep Supervision: 辅助损失 ===
            loss_aux = 0.0
            if aux_outputs is not None and len(aux_outputs) > 0:
                for aux_out in aux_outputs:
                    aux_soft = torch.softmax(aux_out, dim=1)
                    loss_aux_ce = ce_loss(aux_out, label_batch[:].long())
                    loss_aux_dice = dice_loss(aux_soft, label_batch.unsqueeze(1))
                    loss_aux += 0.5 * (loss_aux_dice + loss_aux_ce)
                loss_aux = loss_aux / len(aux_outputs)  # 平均辅助损失
                
                # 总损失 = 主损失 + 0.4 * 辅助损失
                loss = loss_main + 0.4 * loss_aux
            else:
                loss = loss_main
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_main', loss_main, iter_num)
            if aux_outputs is not None:
                writer.add_scalar('info/loss_aux', loss_aux, iter_num)

            # 日志输出
            if aux_outputs is not None and len(aux_outputs) > 0:
                logging.info(
                    'iteration %d : Total_Loss=%.4f | CE=%.4f, Dice_Loss=%.4f(Score=%.3f), Aux_Loss=%.4f [DeepSup OK]' %
                    (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), 
                     1.0 - loss_dice.item(), loss_aux))
            else:
                logging.info(
                    'iteration %d : Total_Loss=%.4f | CE=%.4f, Dice_Loss=%.4f(Score=%.3f)' %
                    (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), 1.0 - loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:
                logging.info('=' * 60)
                logging.info(f'[验证] 开始验证 (iteration {iter_num})...')
                logging.info(f'   验证样本数: {len(db_val)}')
                logging.info(f'   [提示] 验证可能需要几分钟，请耐心等待...')
                model.eval()
                metric_list = 0.0
                # 添加验证进度提示
                val_iterator = tqdm(valloader, desc=f'验证中 (iter {iter_num})', ncols=70, leave=False, 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                for i_batch, sampled_batch in enumerate(val_iterator):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                logging.info('[OK] 验证完成！')
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    # 获取model名称（从snapshot_path提取，已包含register信息）
                    model_save_name = snapshot_path.split('/')[-1]
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(model_save_name))
                    
                    # 保存完整checkpoint（包含训练状态）
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_performance': best_performance,
                        'performance': performance,
                        'mean_hd95': mean_hd95
                    }
                    torch.save(checkpoint, save_mode_path)
                    torch.save(model.state_dict(), save_best)  # 只保存模型权重用于测试

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 根据是否使用 register 添加标识
    use_register = getattr(config.MODEL, 'USE_REGISTER', True)
    num_cls_tokens = getattr(config.MODEL, 'NUM_CLS_TOKENS', 12)
    register_version = getattr(config.MODEL, 'REGISTER_VERSION', 'cross_attn')
    
    # 如果使用 register，在模型名称中添加标识
    model_name = args.model
    if use_register:
        # 根据版本添加不同的后缀
        if register_version == 'cross_attn':
            version_suffix = '_recycle1'  # 方案A/V3
        elif register_version == 'pooled':
            version_suffix = '_recycle'   # 方案D/V2
        elif register_version == 'enhanced_gate':
            version_suffix = '_v4_deepsup'  # V4 + Deep Supervision
        else:
            version_suffix = ''
        model_name = f"{args.model}_register{num_cls_tokens}{version_suffix}"
    
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, model_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # 打印保存路径信息
    print(f"\n{'='*60}")
    print(f"Register机制: {'启用' if use_register else '未启用'}")
    if use_register:
        print(f"Register tokens数量: {num_cls_tokens}")
        # 版本名称映射
        version_names = {
            'cross_attn': "方案A/V3 (Cross-Attention)",
            'pooled': "方案D/V2 (Pooled Aggregation)",
            'enhanced_gate': "方案V4 (Enhanced Gated Fusion - 最优)"
        }
        version_name = version_names.get(register_version, register_version)
        print(f"Register版本: {register_version} ({version_name})")
    print(f"模型保存路径: {snapshot_path}")
    print(f"{'='*60}\n")
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
