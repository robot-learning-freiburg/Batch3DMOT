#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2022
@author: Martin Buechner, mbuechner@cs.uni-freiburg.de
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5, 6, 7"

import glob
from tqdm import tqdm
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime


import torch
import torch.nn as nn
import torchvision.transforms
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.distributed
import torch.utils.data

import torch.distributed
import torch.multiprocessing
import torch.nn.parallel
import torch.optim

import batch_3dmot.utils.dataset
from batch_3dmot.utils.config import ParamLib
#from batch_3dmot.utils.load_scenes import load_scene_meta_list

import batch_3dmot.models.resnet_fully_conv


# ----------- PARAMETER SOURCING --------------

parser = argparse.ArgumentParser()

# General parameters (namespace: main)
parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--version', type=str, help="define the dataset version that is used")

# Specific arguments (namespace: resnet)
parser.add_argument('--batch_size', type=int, help="specify the batch size used for training")
parser.add_argument('--lr', type=float, help="specify the learning rate used for training")
parser.add_argument('--num_epochs', type=int, help="specify the number of epochs")
parser.add_argument('--res_size', type=int, help="specify the image resolution used")
parser.add_argument('--shuffle_data', type=bool, help="define whether to shuffle the provided data")
parser.add_argument('--workers', type=int, help='number of data loading workers')
parser.add_argument('--checkpoint', type=str, help='model path')
parser.add_argument('--save_images', type=bool, help="Declare whether images and reconstructions shall be saved.")

opt = parser.parse_args()

params = ParamLib(opt.config)
params.main.overwrite(opt)
params.resnet.overwrite(opt)


# ---------------------- I/O-SETTINGS ----------------------

# Use this lambda function to convert variable names to strings for print statements
blue = lambda x: '\033[94m' + x + '\033[0m'

print("Used Seed:", params.resnet.manual_seed)
random.seed(params.resnet.manual_seed)
torch.manual_seed(params.resnet.manual_seed)

# interval for displaying training info
log_interval = 10

# using bilinear interpolation (interpolation=2 -> bilinear)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize([params.resnet.res_size,
                                                                           params.resnet.res_size]),
                                            torchvision.transforms.ToTensor(),
                                            ])



# ---------------------- TRAINING / VALIDATION ----------------------


def loss_function(recon_x, x):
    """
    Define some particular loss function to be used when training the ResNet-like autoencoder structure.
    Args:
        recon_x:
        x:

    Returns:

    """
    loss_fct = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    return loss_fct


def imshow(img):
    """

    Args:
        img:

    Returns:

    """
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # run through: imshow(torchvision.utils.make_grid(images))

batch_3dmot.utils.dataset.check_mkdir(os.path.join(params.paths.models, 'resnet/'))
class_dict_used = batch_3dmot.utils.dataset.get_class_config(params=params, class_dict_name=params.main.class_dict)

def init_process(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def distrib_dataloader(rank: int, world_size: int, split_name: str):
    
    
    dataset = batch_3dmot.utils.dataset.ImageDataset(params=params,
                                                     class_dict=class_dict_used,
                                                     split_name=split_name,
                                                     transform=transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=params.resnet.batch_size, 
                                             shuffle=False, 
                                             sampler=sampler,
                                             num_workers=8)
    print("Initializing dataloader for rank", rank)

    return dataloader

def get_model():
    return batch_3dmot.models.resnet_fully_conv.ResNetAE()

def train_distributed(rank: int, world_size: int, params: batch_3dmot.utils.config.ParamLib) -> None:
    init_process(rank=rank, world_size=world_size)
    print("RANK", rank, "in WORLD N =", world_size, " - Process initialized.")

    # First run the model+dataloader on rank 0 device.
    if rank == 0:
        distrib_dataloader(rank, world_size, 'train')
        get_model()
    torch.distributed.barrier()
    print(f"Rank {rank}/{world_size} training process passed data download barrier.\n")

    model = get_model()
    model.cuda(rank)
    model.train()

    print("initialized model for rank", rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    train_dataloader = distrib_dataloader(rank=rank, world_size=world_size, split_name='train')
    val_dataloader = distrib_dataloader(rank=rank, world_size=world_size, split_name='val')
    
    num_batches_train = len(train_dataloader.dataset) / params.resnet.batch_size
    num_batches_val = len(val_dataloader.dataset) / params.resnet.batch_size

    criterion = torch.nn.MSELoss().cuda(rank)
    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=params.resnet.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=params.resnet.scheduler_step_size,
                                                gamma=params.resnet.scheduler_gamma)

    # Set up a tensorboard writer
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(params.paths.models,
                                                'runs/resnet/resnet_%s_%s_%s_%s_%s_%s'
                                                % (str(datetime.datetime.now()).split('.')[0],
                                                   params.main.version,
                                                   params.main.class_dict,
                                                   params.resnet.batch_size,
                                                   params.resnet.lr,
                                                   params.resnet.num_epochs)))
    if rank == 0:
        global_train_step = 0
        global_val_step = 0

    for epoch in range(params.resnet.num_epochs):
        
        if rank == 0:
            metrics = defaultdict(list)

        for i, data in enumerate(train_dataloader):

            optimizer.zero_grad()

            images, labels = data
            # Move data to respective GPU instance
            images, labels = images.cuda(rank), labels.cuda(rank).view(-1,)

            reconstructions = model(images)
            loss = criterion(reconstructions, images) / params.resnet.batch_size

            loss.backward()
            optimizer.step()

            if rank == 0:
                metrics['train/ep_loss'].append(loss.item())
                writer.add_scalar("Loss/train", loss, global_train_step)

                if (i + 1) % 10 == 0:
                    tqdm.write('GPU: %d [%d: %d/%d] %s loss: %f' % (rank, epoch, i, num_batches_train/world_size, blue('train'), loss.item()))
            
                global_train_step += 1

        with torch.no_grad():
            model.eval()
            
            for j, data in enumerate(val_dataloader):
  
                images, labels = data
                # Move data to respective GPU instance
                images, labels = images.cuda(rank), labels.cuda(rank).view(-1,)

                reconstructions = model(images)
                loss = criterion(reconstructions, images) / params.resnet.batch_size

                                      

                if params.resnet.save_images_val_test and j % 10 == 0:
                    batch_3dmot.utils.dataset.check_mkdir(os.path.join(params.paths.preprocessed_data, 'reconst_img/'))
                    torchvision.utils.save_image(images,
                                                 params.paths.preprocessed_data
                                                 + 'resnet_training80/val_inputs'
                                                 + str(epoch) + '_'
                                                 + str(j) + '.png')

                    torchvision.utils.save_image(reconstructions,
                                                 params.paths.preprocessed_data
                                                 + 'resnet_training80/val_outputs_'
                                                 + str(epoch) + '_'
                                                 + str(j) + '.png')
                if rank == 0:
                    writer.add_scalar("Loss/val", loss, global_val_step)
                    metrics['val/ep_loss'].append(loss.item())  

                    global_val_step += 1

                if (j + 1) % log_interval == 0:
                    tqdm.write('[%d: %d] %s loss: %f ' % (epoch, j, blue('val'), loss.item()))

        if rank == 0:
            metrics = {k: np.nanmean(v) for k, v in metrics.items()}
            
            writer.add_scalar("ep_loss/train", metrics['train/ep_loss'], epoch)
            writer.add_scalar("ep_loss/val", metrics['val/ep_loss'], epoch)
            scheduler.step()

            writer.flush()
            torch.save(model.module.state_dict(),
                       '%s/resnet80eps_epoch%d_%s_%s.pth' % (os.path.join(params.paths.models, 'resnet/'),
                                                        epoch,
                                                        params.main.version,
                                                        params.main.class_dict))
    if rank == 0:                                 
        writer.close()
                                     

if __name__ == '__main__':
    #train(model=resnet_ae, optimizer=optimizer, scheduler=scheduler, device=device, writer=resnet_writer,
    #      train_loader=train_loader, val_loader=val_loader,
    #      log_interval=10, save_images=params.resnet.save_images)

    ########################### MULTI-GPU SETTING ###################################
    world_size = 4

    torch.multiprocessing.spawn(train_distributed, args=(world_size, params), nprocs=world_size, join=True)

    print('Finished Training')
    print('Saving Model...')

