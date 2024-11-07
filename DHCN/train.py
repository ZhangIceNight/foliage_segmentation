import os
import torch
import argparse
import random
import numpy as np
from solver.Solver import Model_Solver

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# seed
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def main(config):
    folder_path = {
        'LabelledPC_dataset': 'your/dataset/path/',
    }
    IoU_all = np.zeros(config.train_test_num, dtype=np.float64)
    Acc_all = np.zeros(config.train_test_num, dtype=np.float64)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))

        solver = Model_Solver(config, folder_path[config.dataset])
        IoU_all[i], Acc_all[i] = solver.train()
        IoU_med = np.median(IoU_all)
        Acc_med = np.median(Acc_all)
        print('Testing median IoU %4.4f,\tmedian Acc %4.4f' % (IoU_med, Acc_med))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='LabelledPC', help='')
    parser.add_argument('--resume', dest='resume', type=bool, default=False, help='')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs for training')
    parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--model_name', dest='model_name', type=str, default="DHCN", help='')
    config = parser.parse_args()
    main(config)


