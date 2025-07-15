import argparse
import os
from data_utils.LeafDataLoader import LeafDatasetWholeScene
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import wandb
from timm.scheduler import CosineLRScheduler
from models import Point_NN_Seg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['non-leaf', 'leaf']

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=300, type=int, help='Epoch to run [default: 300]')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--ckpts', type=str, default='pretrain.pth', help='ckpts')
    
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--dim', type=int, default=144)
    parser.add_argument('--k', type=int, default=90)
    parser.add_argument('--de_k', type=int, default=6)
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)
    parser.add_argument('--gamma', type=int, default=300)
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/'
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = LeafDatasetWholeScene(root=root, split='train', block_points=NUM_POINT)
    print("start loading test data ...")
    TEST_DATASET = LeafDatasetWholeScene(root=root, split='test', block_points=NUM_POINT)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, 
                                                batch_size=BATCH_SIZE, 
                                                shuffle=True,
                                                worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=False)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    MODEL_NN = importlib.import_module('Point_NN_Seg')
    print("model loaded")
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    print("model copied")
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print("model applied")


    @torch.no_grad()
    def build_memory_bank(train_loader, point_nn, args):
        """构建memory bank (按类别计算原型)"""
        feature_memory = []
        label_memory = []
        
        point_nn.eval()
        with torch.no_grad():
            for points, target in tqdm(train_loader, desc='Building memory bank'):
                points = points.float().cuda()
                points = points.transpose(2, 1)  # [B, 3, N]

                # 获取特征
                point_features = point_nn(points)
                point_features = point_features.permute(0, 2, 1)  # [B, N, C]
                
                # 按batch和类别处理
                for b in range(points.shape[0]):
                    for class_idx in range(NUM_CLASSES):
                        # 获取当前类别的mask
                        class_mask = (target[b] == class_idx)
                        if torch.sum(class_mask) == 0:
                            continue
                    
                        # 计算类别原型
                        class_features = point_features[b][class_mask]  # [N_class, C]
                        class_prototype = class_features.mean(0).unsqueeze(0)  # [1, C]
                    
                        # 保存原型和标签
                        feature_memory.append(class_prototype)
                        label_memory.append(torch.tensor([class_idx], device='cuda'))
    
        # 合并所有原型和标签
        feature_memory = torch.cat(feature_memory, dim=0)  # [M, C]
        feature_memory = F.normalize(feature_memory, dim=-1)
        feature_memory = feature_memory.t()  # [C, M]
        
        # 转换标签为one-hot
        label_memory = torch.cat(label_memory, dim=0)  # [M]
        label_memory = F.one_hot(label_memory, num_classes=NUM_CLASSES).float()  # [M, NUM_CLASSES]
    
        return feature_memory, label_memory


    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        num_trainable_params = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
                num_trainable_params += param.numel()
            else:
                decay.append(param)
                num_trainable_params += param.numel()

        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        log_string('########################################################################')
        log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
            '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
        log_string('########################################################################')

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    
    # load checkpoint
    start_epoch = 0
    if args.ckpts is not None:
        if args.ckpts[:13] == "segmentation/":
            args.ckpts = args.ckpts[13:]
        classifier.load_model_from_ckpt(args.ckpts)
        log_string('Load model from %s' % args.ckpts)
    else:
        log_string('No existing model, starting training from scratch...')

    # try:
    #     checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0


    param_groups = add_weight_decay(classifier, weight_decay=0.05)

    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.decay_rate)

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)
    
    
    print("optimizer created")

    global_epoch = 0
    best_iou = 0
    best_acc = 0
  

    # 初始化wandb
    wandb.init(
        project="pointnet-leaf-seg",
        name=args.log_dir,
        config={
            "model": args.model,
            "batch_size": args.batch_size,
            "num_point": args.npoint,
            "learning_rate": args.learning_rate,
            "epochs": args.epoch,
            "optimizer": args.optimizer
        }
    )
    print("wandb initialized")
    classifier.zero_grad()
    

    # load point_nn model
    point_nn = MODEL_NN.Point_NN_Seg(input_points=args.npoint, 
                           num_stages=args.stages,
                           embed_dim=args.dim, 
                           k_neighbors=args.k, 
                           de_neighbors=args.de_k,
                           alpha=args.alpha, 
                           beta=args.beta).cuda()
    point_nn.eval()
    
    # build memory bank
    print("building memory bank...")
    feature_memory, label_memory = build_memory_bank(trainDataLoader, point_nn, NUM_CLASSES)





    # Start training
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))


        loss_batch = []
        mean_correct = []
        classifier = classifier.train()
        num_iter = 0
        '''learning one epoch'''
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            num_iter += 1
            points = points.data.numpy()
            # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            # point_nn part
            with torch.no_grad():
                point_features = point_nn(points)
                point_features = point_features.permute(0, 2, 1)  # [B, N, C]
                point_features = F.normalize(point_features, dim=-1)

                batch_size, num_points = point_features.shape[:2]
                point_features = point_features.reshape(-1, point_features.shape[-1])

                # 计算相似度
                similarity = point_features @ feature_memory
                nn_logits = (-args.gamma * (1 - similarity)).exp() @ label_memory
                nn_logits = nn_logits.reshape(batch_size, num_points, NUM_CLASSES)
            
            # mamba_part
            seg_logits = classifier(points)
            seg_logits_fusion = seg_logits + nn_logits

            seg_pred = F.log_softmax(seg_logits_fusion, dim=1).contiguous().view(-1, NUM_CLASSES)
            
            seg_logits_fusion = seg_logits_fusion.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1)
            loss = criterion(seg_logits_fusion, target)

            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            if num_iter == 1:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()


        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        train_acc = np.mean(mean_correct)
        train_loss = np.mean(loss_batch)
        log_string('Training mean loss: %.5f' % train_loss)
        log_string('Training accuracy: %.5f' % train_acc)

        # 记录训练指标
        wandb.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=global_epoch)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            loss_batch = []
            mean_correct = []
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            
            classifier = classifier.eval()
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                #load data
                points = points.data.numpy() # [B, N, 3]
                points = torch.Tensor(points) # [B, N, 3]
                points, target = points.float().cuda(), target.long().cuda() # [B, N, 3]
                points = points.transpose(2, 1) # [B, 3, N]

                # point_nn part
                point_features = point_nn(points)
                point_features = point_features.permute(0, 2, 1)  # [B, N, C]
                point_features = F.normalize(point_features, dim=-1)

                batch_size, num_points = point_features.shape[:2]
                point_features = point_features.reshape(-1, point_features.shape[-1])

                # 计算相似度
                similarity = point_features @ feature_memory
                nn_logits = (-args.gamma * (1 - similarity)).exp() @ label_memory
                nn_logits = nn_logits.reshape(batch_size, num_points, NUM_CLASSES)                
                
                # mamba part
                seg_logits = classifier(points)
                seg_logits_fusion = seg_logits + nn_logits

                seg_pred = F.log_softmax(seg_logits_fusion, dim=1).contiguous().view(-1, NUM_CLASSES)
                seg_logits_fusion = seg_logits_fusion.contiguous().view(-1, NUM_CLASSES)
                target = target.view(-1) # [B*N]

                #loss
                loss = criterion(seg_logits_fusion, target)
                loss_batch.append(loss.detach().cpu())

                #accuracy

                ## get max class index        
                pred_choice = seg_pred.data.max(1)[1] # [B*N, 2] -> [B*N]
                ## compare batch accuracy
                correct = pred_choice.eq(target.data).cpu().sum()   
                mean_correct.append(correct.item() / (args.batch_size * args.npoint))

                
                ## pred and target to numpy
                pred_val = pred_choice.contiguous().cpu().data.numpy() # [B*N, 2]
                target = target.cpu().data.numpy() # [B*N]
                ## 计算每个类别的指标
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((target == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (target == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (target == l)))

            # 计算平均指标
            eval_loss = np.mean(loss_batch)
            test_overall_accuracy = np.mean(mean_correct)
            class_acc = np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6)
            class_iou = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
            mean_iou = np.mean(class_iou)
            
            # 记录评估指标
            wandb.log({
                "eval/loss": eval_loss,
                "eval/accuracy": test_overall_accuracy,
                "eval/mean_iou": mean_iou,
            }, step=global_epoch)

            # 记录每个类别的指标
            for i in range(NUM_CLASSES):
                wandb.log({
                    f"eval/class_{classes[i]}_iou": class_iou[i],
                    f"eval/class_{classes[i]}_acc": class_acc[i]
                }, step=global_epoch)

            log_string('test mean loss: %.5f' % eval_loss)
            log_string('test accuracy: %.5f' % test_overall_accuracy)
            log_string('test mean IoU: %.5f' % mean_iou)

            if mean_iou >= best_iou:
                best_iou = mean_iou
                best_acc = test_overall_accuracy
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'mean_iou': best_iou,
                    'accuracy': best_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                # log_string('Saving model....')
            log_string('Best accuracy: %.5f' % best_acc)
            log_string('Best mIoU: %.5f' % best_iou)
        global_epoch += 1

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    main(args)
