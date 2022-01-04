import argparse
import logging
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

# by DM
#from nsml import HAS_DATASET, DATASET_PATH
from arguments_fm import parser
from utils.load_dataset_fm import load_datasets
from dataset.randaugment import RandAugmentMC

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN
import torchvision.transforms as T

#logger = logging.getLogger(__name__)
best_acc = 0


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode='constant')]) #reflect
        self.strong = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode='constant'), #]) #reflect
            RandAugmentMC(n=2, m=10)])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class TransformFixMatch2(object):
    def __init__(self, mean, std):
        self.weak = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode='constant')]) #reflect
        self.strong = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode='constant'), #]) #reflect
            RandAugmentMC(n=2, m=10)])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class MyRandomImages(Dataset):
    def __init__(self, file_path, transform=None, data_num=50000, exclude_cifar=True):
        self.transform = transform
        self.data = np.load(file_path).astype(np.uint8)

        if data_num != -1:
            all_id = list(range(len(self.data)))
            sample_id = random.sample(all_id, data_num)
            self.data = self.data[sample_id]

        #from torchvision.utils import save_image
        #save_image(self.transform(self.data[100])[0], 'test_ood.png')

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, 0 , index  # 0 is the class

    def __len__(self):
        return len(self.data)

class MySVHN(Dataset):
    def __init__(self, file_path, download, transform):
        self.svhn = SVHN(file_path, download=download, transform=transform, split='extra') #, split='extra'

    def __getitem__(self, index):
        data, target = self.svhn[index]
        return data, target, index

    def __len__(self):
        return len(self.svhn)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps,num_cycles=7./16.,last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def main():
    #print("NSML: ", DATASET_PATH, HAS_DATASET)

    args = parser.parse_args()
    print(args.local_rank)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
        args.batch_size = int(args.batch_size/args.world_size)

    args.device = device
    print("args: ", args)

    print(f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    if args.seed is not None:
        set_seed(args)

    global best_acc

    if not False: #HAS_DATASET:
        args.in_data_name = 'cifar10_dm' #cifar10_dm, CIFAR100
        file_path = file_path = '../data/cifar10/'

        args.ood_data_name = '300K_Random_dm' #svhn_dm_extra, 300K_Random_dm
        file_path_ood = '../data/300K_Random/' #'../data/svhn/', '../data/300K_Random/'
    #else:
        #file_path = os.path.join(DATASET_PATH[0], 'train')
        #in_data_name = file_path.split('/')[2]

        #file_path_ood = os.path.join(DATASET_PATH[1], 'train')
        #ood_data_name = file_path_ood.split('/')[2]
        #print("ood_data_name: ", ood_data_name)

    # In distribution
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, file_path)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(labeled_dataset,batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # OOD
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    if args.ood_data_name =='svhn_dm_extra':
        train_ood = MySVHN(file_path_ood, download=True, transform=TransformFixMatch(mean=cifar10_mean, std= cifar10_std))
    elif args.ood_data_name == '300K_Random_dm':

        #NSML
        print(os.listdir(file_path_ood))
        filename = os.listdir(file_path_ood)[0] #[1] for NSML
        f_path = file_path_ood + '/' + filename
        print(f_path)
        train_ood = MyRandomImages(f_path, transform=TransformFixMatch2(mean=cifar10_mean, std= cifar10_std))

    ood_trainloader = DataLoader(train_ood, batch_size=args.batch_size, pin_memory=True)
    # unlabeled
    if args.ood_rate == 0:
        len_in = len(unlabeled_dataset)
        unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size * args.mu, shuffle=True, num_workers=0, drop_last=True)
    else:
        len_in = len(unlabeled_dataset)
        len_ood = int(len(unlabeled_dataset) * args.ood_rate)
        train_ood_unlabeled, _ = torch.utils.data.random_split(train_ood, [len_ood, len(train_ood) - len_ood])
        datalist = [unlabeled_dataset, train_ood_unlabeled]

        multi_datasets = torch.utils.data.ConcatDataset(datalist)
        unlabeled_trainloader = DataLoader(multi_datasets, batch_size=args.batch_size * args.mu, shuffle=True, num_workers=0, drop_last=True)
        print("# in: {}, # ood: {} ".format(len(unlabeled_dataset), len(train_ood_unlabeled)))

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet_fm as models
            model = models.build_wideresnet(depth=args.model_depth, widen_factor=args.model_width,
                                            dropout=0, num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality, depth=args.model_depth,
                                         width=args.model_width, num_classes=args.num_classes)
        print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    torch._C._broadcast_coalesced
    model = create_model(args).to(args.device)
    #model = nn.DataParallel(model)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.num_labeled}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size per GPU = {args.batch_size}")
    print(f"  Total train batch size = {args.batch_size*args.world_size}")
    print(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, ood_trainloader,
          model, optimizer, ema_model, scheduler, len_in)

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, ood_trainloader,
          model, optimizer, ema_model, scheduler, len_in):

    global best_acc
    test_accs = []
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    ood_iter = iter(ood_trainloader)

    model.train()
    t_prev = time.time()
    t_prev_epoch = time.time()
    log = []
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        pseudo_acc = AverageMeter()
        for batch_idx in range(args.eval_step):

            t1 = time.time()
            try:
                inputs_x, targets_x, _ = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, _ = labeled_iter.next()

            t2 = time.time()

            try:
                (inputs_u_w, inputs_u_s), targets_u, indexs = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_u, indexs = unlabeled_iter.next()

            t3 = time.time()

            # ODNL
            if args.reg_weight > 0:
                try:
                    (inputs_ood_u, inputs_ood_u2), _, _ = ood_iter.next()
                except:
                    ood_iter = iter(ood_trainloader)
                    (inputs_ood_u, inputs_ood_u2), _, _ = ood_iter.next()

            t4 = time.time()
            #print(t2-t1, t3-t2, t4-t3)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, preds_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, preds_u, reduction='none') * mask).mean()
            loss = Lx + args.lambda_u * Lu

            # ODNL
            if epoch>10 and args.reg_weight > 0:
                outputs_ood = model(inputs_ood_u.to(args.device))
                target_random = torch.LongTensor(outputs_ood.shape[0]).random_(0, args.n_class).to(args.device)

                ce = nn.CrossEntropyLoss(reduction='none')
                target_loss = ce(outputs_ood, target_random)
                reg_loss = torch.sum(target_loss) / target_loss.size(0)
                loss = loss + args.reg_weight * reg_loss
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            correct, total = 0, 0
            for i, idx in enumerate(indexs):
                if idx < len_in:
                    correct += (preds_u[i]==targets_u[i].to(args.device)).item()
                    total += 1
            #print(correct, total)
            #print(correct/total)
            pseudo_acc.update(correct/total)

            if batch_idx == 0: #args.eval_step-1:
                print("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
                      "Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. PseudoAcc: {acc:.2f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    #lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg,
                    acc=pseudo_acc.avg))

                t_elapsed = time.time() - t_prev
                t_prev = time.time()
                print('Elapsed time (sec): {:.2f}'.format(t_elapsed))

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            best_acc = max(test_acc, best_acc)
            '''
            model_to_save = ALOOD_OURS.module if hasattr(ALOOD_OURS, "module") else ALOOD_OURS
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            '''
            test_accs.append(test_acc)
            print('Best top-1 acc: {:.2f}, Mean top-1 acc: {:.2f}, Curr top-1 acc: {:.2f}'
                  .format(best_acc, np.mean(test_accs[-20:]), test_acc))
            #print('Mean top-1 acc: {:.2f}'.format(np.mean(test_accs[-20:])))
            #print('Curr top-1 acc: {:.2f}\n'.format(np.mean(test_acc)))
            log.append([test_acc, np.mean(test_accs[-20:]), best_acc, pseudo_acc.avg])

            t_elapsed = time.time()-t_prev_epoch
            t_prev_epoch = time.time()
            print('Elapsed time (sec): {:.2f}\n'.format(t_elapsed))

        if epoch % 100 == 0:
            folder_path = 'result/'
            file_name = args.in_data_name+'_'+args.ood_data_name+'_ood_rate'+str(args.ood_rate)+'_n_labeled'+str(args.num_labeled)+'_fixmatch_v1.csv'
            #print(file_name)
            print(np.array(log))
            np.savetxt(folder_path+file_name, np.array(log), fmt='%.4f', delimiter=',')

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
        if not args.no_progress:
            test_loader.close()

    print("epoch: {}, top-1 acc: {:.2f}, top-5 acc: {:.2f}".format(epoch+1, top1.avg, top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()