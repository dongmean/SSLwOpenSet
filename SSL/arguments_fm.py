import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'],
                    help='dataset name')
parser.add_argument('--ood-rate', type=float, default=0)
parser.add_argument('--num-labeled', type=int, default=250,
                    help='number of labeled data')
parser.add_argument('--reg-weight', type=int, default=0,
                    help='regularization')
parser.add_argument('--n-class', type=int, default=10)
parser.add_argument("--expand-labels", action="store_true",
                    help="expand labels to fit eval steps")
parser.add_argument('--arch', default='wideresnet', type=str,
                    choices=['wideresnet', 'resnext'],
                    help='dataset name')
parser.add_argument('--total-steps', default=2 ** 20, type=int,
                    help='number of total steps to run')
parser.add_argument('--eval-step', default=1024, type=int,
                    help='number of eval steps to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='train batchsize')
parser.add_argument('--test-batch-size', default=1000, type=int,
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    help='initial learning rate')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')
parser.add_argument('--use-ema', action='store_true', default=True,
                    help='use EMA ALOOD_OURS')
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--out', default='result',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help="random seed")
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt_level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")

args = parser.parse_args()

