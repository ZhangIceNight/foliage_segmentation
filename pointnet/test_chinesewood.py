"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ChineseWoodLoader import ChineseWoodDataset
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
# import pointnet.data_utils.provider as provider
import numpy as np
import torch.nn.functional as F



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['non-leaf', 'leaf']
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True           




def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def save_single_visual_result(pred_label, target, save_path):
    # 保存预测结果和真实标签到文件 filename_pred.txt 和 filename_gt.txt
    pred_path = save_path.replace('.npy', '_pred.txt')
    gt_path = save_path.replace('.npy', '_gt.txt')
    with open(pred_path, 'w') as f:
        for i in range(len(pred_label)):
            f.write(f"{pred_label[i]}\n")
    with open(gt_path, 'w') as f:
        for i in range(len(target)):
            f.write(f"{target[i]}\n")

def save_batch_visual_result(pred_labels, targets, save_paths):
    # 保存预测结果和真实标签到文件 filename_pred.txt 和 filename_gt.txt
    """
    Usage:
    pred_labels = np.array([[1, 2, 3], [4, 5, 6]])
    targets = np.array([[1, 2, 3], [4, 5, 6]])
    save_paths = ['path/to/save/1.npy', 'path/to/save/2.npy']
    save_batch_visual_result(pred_labels, targets, save_paths)
    """
    B = len(pred_labels)
    for i in range(B):
        save_single_visual_result(pred_labels[i], targets[i], save_paths[i])



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # get experiment_dir
    if args.log_dir is None:
        raise ValueError("log_dir is required")
    else:
        experiment_dir = Path(args.log_dir)

    # get visual_dir
    visual_dir = experiment_dir.joinpath('visual/')
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.npoint

    root = 'data_chinesewood/'

    print("start loading test data ...")
    TEST_DATASET = ChineseWoodDataset(root=root, 
                                    split='test',
                                    block_points=NUM_POINT)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=False)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    print("model loaded")
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    classifier.apply(inplace_relu)
    print("model applied")
    # load checkpoint # example: log/sem_seg_chinesewood/pt_hmamba_2024-12-13_16-11/checkpoints/model_best.pth
    if args.ckpts is not None:
        ckpt = torch.load(args.ckpts)
        classifier.load_state_dict(ckpt['model_state_dict'], strict=False)
        log_string('Load model from %s' % args.ckpts)
    else:
        raise ValueError("ckpts is required")
    classifier = classifier.eval()

    with torch.no_grad():
        log_string('---- EVALUATION ----')
        file_list = TEST_DATASET.file_list
        for batch_idx, (points, labels) in tqdm(enumerate(testDataLoader)):
            points, labels = points.cuda(), labels.cuda() # [B, N, 3] [B, N]
            points = points.transpose(2, 1) # [B, 3, N]
            seg_pred = classifier(points) # [B, N, NUM_CLASSES]

            seg_pred_soft = F.log_softmax(seg_pred, dim=1) # [B, N, NUM_CLASSES]
            pred_choice = seg_pred_soft.data.max(1)[1] # [B, N]

            # filename example: [tree1.npy tree2.npy ...]
            # visual_dir example: ./log/sem_seg_chinesewood/visual/
            log_string(f"Saving visual result to {visual_dir} ...")
            save_path = os.path.join(visual_dir, file_list[batch_idx])
            save_batch_visual_result(pred_choice, labels, save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
