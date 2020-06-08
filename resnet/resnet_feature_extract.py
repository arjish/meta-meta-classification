import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet.resnet_v2  as resnet
import numpy as np


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original_tuple and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


model_names = sorted(name for name in resnet.__dict__
                     if name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extraction')
parser.add_argument('-d', '--data', metavar='DIR', default='',
    help='path to dataset')
parser.add_argument('-r', '--results_folder', metavar='RES_DIR', default='extracted_features',
    help='path to the results')
parser.add_argument('-f', '--imageFolderName', metavar='FOLDER', default='train')
parser.add_argument('-a', '--arch', default='resnet152', choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
         ' (default: resnet152)')
parser.add_argument('-j', '--workers', default=4, type=int,
    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
    help='mini-batch size (default: 32), this is the total '
         'batch size of all GPUs on the current node when '
         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--resume', default='checkpoint.pth.tar', type=str, metavar='PATH',
    help='path to latest checkpoint (default: checkpoint.pth.tar)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
    help='Use multi-processing distributed training to launch '
         'N processes per node, which has N GPUs. This is the '
         'fastest way to use PyTorch for either single node or '
         'multi node data parallel training')


def createFolderStructure(args):
    results_path = os.path.join(args.data, args.results_folder)
    imageFolderName = args.imageFolderName
    data_path = os.path.join(args.data, imageFolderName)
    classFolders_list = [label \
                         for label in os.listdir(data_path) \
                         if os.path.isdir(os.path.join(data_path, label))]
    for folder_name in classFolders_list:
        if not os.path.exists(os.path.join(results_path, imageFolderName, folder_name)):
            os.makedirs(os.path.join(results_path, imageFolderName, folder_name))


best_acc1 = 0


def main():
    args = parser.parse_args()

    createFolderStructure(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url=="env://" and args.world_size==-1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    imageFolderName = args.imageFolderName
    imagedir = os.path.join(args.data, imageFolderName)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url=="env://" and args.rank==-1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # reconstruct the classifier once the model is instantiated.
    # remove last fully-connected layer.
    # use model.module to get the model (ResNet) when wrapped with DataParallel.
    if args.gpu is None:
        model.module = nn.Sequential(*list(model.module.children())[:-1])
    else:
        model = nn.Sequential(*list(model.children())[:-1])

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    data_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(imagedir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    extract_features(args, data_loader, model)


def extract_features(args, data_loader, model):
    imageFolderName = args.imageFolderName
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, _, image_path in data_loader:
            # compute output
            output_tensor = model(input.cuda()).cpu()  # convert the input to cuda before feeding
            output = np.squeeze(output_tensor.detach().numpy())

            for i in range(output.shape[0]):
                root, image_name = os.path.split(image_path[i])
                root, folder_name = os.path.split(root)
                save_path = os.path.join(args.data, args.results_folder, imageFolderName, folder_name)
                np.save(os.path.join(save_path, image_name.split('.')[0]), output[i])


if __name__=='__main__':
    main()
