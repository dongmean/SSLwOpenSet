import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from tqdm import tqdm
from utils import AverageMeter, ova_loss,\
    save_checkpoint, ova_ent\

from utils.misc_fm import test, test_ood, exclude_dataset
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0

def train(args, labeled_trainloader, unlabeled_dataset, test_loader,
          model, optimizer, ema_model, scheduler):
    #if args.amp:
    #    from apex import amp

    global best_acc
    global best_acc_val

    log = []
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    pseudo_acc = AverageMeter()
    end = time.time()


    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak


    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler


    for epoch in range(args.start_epoch, args.epochs):
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        if epoch < args.start_fix:
            unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                               sampler = train_sampler(unlabeled_dataset),
                                               batch_size = args.batch_size * args.mu,
                                               num_workers = args.num_workers,
                                               drop_last = True)
            unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                               sampler=train_sampler(unlabeled_dataset_all),
                                               batch_size=args.batch_size * args.mu,
                                               num_workers=args.num_workers,
                                               drop_last=True)

        if epoch >= args.start_fix:
            ## pick pseudo-inliers
            ind_selected = exclude_dataset(args, unlabeled_dataset, ema_model.ema)

            unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                               sampler=SubsetRandomSampler(ind_selected),
                                               batch_size=args.batch_size * args.mu,
                                               num_workers=args.num_workers,
                                               drop_last=True)
            unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                                   sampler=SubsetRandomSampler(ind_selected),
                                                   batch_size=args.batch_size * args.mu,
                                                   num_workers=args.num_workers,
                                                   drop_last=True)

        unlabeled_iter = iter(unlabeled_trainloader)
        unlabeled_all_iter = iter(unlabeled_trainloader_all)

        for batch_idx in range(args.eval_step):
            ## Data loading

            try:
                (_, inputs_x_s, inputs_x), targets_x, _ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x, _ = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s, _), targets_u, indices = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), targets_u, indices = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s, _), _, _ = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, _), _, _ = unlabeled_all_iter.next()
            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]

            inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
            inputs = torch.cat([inputs_x, inputs_x_s,
                                inputs_all], 0).to(args.device)
            targets_x = targets_x.to(args.device)
            ## Feed data
            logits, logits_open = model(inputs)
            logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)

            ## Loss for labeled samples
            Lx = F.cross_entropy(logits[:2*b_size],
                                      targets_x.repeat(2), reduction='mean')
            Lo = ova_loss(logits_open[:2*b_size], targets_x.repeat(2))

            ## Open-set entropy minimization
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.

            ## Soft consistenty regularization
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1 = F.softmax(logits_open_u1, 1)
            logits_open_u2 = F.softmax(logits_open_u2, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1 - logits_open_u2)**2, 1), 1))

            if epoch >= args.start_fix:
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
                logits, logits_open_fix = model(inputs_ws)
                logits_u_w, logits_u_s = logits.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, preds_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                L_fix = (F.cross_entropy(logits_u_s,
                                         preds_u,
                                         reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())

            else:
                L_fix = torch.zeros(1).to(args.device).mean()
            loss = Lx + Lo + args.lambda_oem * L_oem  \
                   + args.lambda_socr * L_socr + L_fix
            #if args.amp:
            #    with amp.scale_loss(loss, optimizer) as scaled_loss:
            #        scaled_loss.backward()
            #else:
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

            if epoch >= args.start_fix:
                correct, total = 0, 0
                for i, idx in enumerate(indices):
                    if idx < 50000: #argument?
                        correct += (preds_u[i] == targets_u[i].to(args.device)).item()
                        total += 1
                    # else:
                    #    print(idx)

                # print(correct, total)
                # print(correct/total)
                pseudo_acc.update(correct / total)

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            test_acc = test(args, test_loader, test_model, epoch)
            test_accs.append(test_acc)
           # for ood in ood_loaders.keys():
           #     roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
           #     logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            '''
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            '''
            print('Best top-1 acc: {:.2f}, Mean top-1 acc: {:.2f}, Curr top-1 acc: {:.2f}'
                  .format(best_acc, np.mean(test_accs[-20:]), test_acc))

            log.append([test_acc, np.mean(test_accs[-20:]), best_acc, pseudo_acc.avg])

            if epoch % 100 == 0:
                folder_path = 'result/'
                file_name = args.dataset + '_' + args.ood_data_name + '_ood_rate' + str(
                    args.ood_rate) + '_n_labeled' + str(args.num_labeled) + '_openmatch_v1.csv'
                # print(file_name)
                print(np.array(log))
                np.savetxt(folder_path + file_name, np.array(log), fmt='%.4f', delimiter=',')


    if args.local_rank in [-1, 0]:
        args.writer.close()
