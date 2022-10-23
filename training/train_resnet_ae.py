#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2022
@author: Martin Buechner, buechner@cs.uni-freiburg.de
"""

import os
import glob
from tqdm import tqdm
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import nuscenes


import torch
import torch.nn as nn
import torchvision.transforms
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

import batch_3dmot.utils.dataset
from batch_3dmot.utils.config import ParamLib
from batch_3dmot.utils.load_scenes import load_scene_meta_list

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

# Detect CUDA resources
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
print("Using", torch.cuda.device_count(), "GPU!")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize([params.resnet.res_size,
                                                                           params.resnet.res_size],
                                                                          interpolation=2),
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


if params.main.dataset == 'nuscenes':
    # Load lists with all scene meta data based on the dataset, the version,
    nusc, meta_lists = load_scene_meta_list(data_path=params.paths.data,
                                            dataset=params.main.dataset,
                                            version=params.main.version)

    if params.main.version == "v1.0-mini":
        train_scene_meta_list, val_scene_meta_list = meta_lists
        test_scene_meta_list = None

        train_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, train_scene_meta_list)
        val_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, val_scene_meta_list)
        test_tokens = None

        split_name_train = 'mini_train'
        split_name_val = 'mini_val'
        split_name_test = None

    elif params.main.version == "v1.0-trainval":
        train_scene_meta_list, val_scene_meta_list = meta_lists
        test_scene_meta_list = None
        
        train_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, train_scene_meta_list)
        val_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, val_scene_meta_list)
        #test_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, test_scene_meta_list)

        split_name_train = 'train'
        split_name_val = 'val'
        split_name_test = None

    elif params.main.version == "v1.0-test":
        train_scene_meta_list, val_scene_meta_list, test_scene_meta_list = meta_lists

        train_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, train_scene_meta_list)
        val_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, val_scene_meta_list)
        test_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, test_scene_meta_list)
        split_name_train = 'train'
        split_name_val = 'val'
        split_name_test = 'test'

    else:
        train_scene_meta_list, val_scene_meta_list, test_scene_meta_list = None, None, None
        train_tokens, val_tokens, test_tokens = None, None, None
        split_name_train, split_name_val, split_name_test = None, None, None

    # Define used classes
    class_dict_used = batch_3dmot.utils.dataset.get_class_config(params=params, class_dict_name=params.main.class_dict)
    num_classes = len(class_dict_used)

    print("TOKEN LIST LENGTHS")
    print(len(train_tokens))
    print(len(val_tokens))
    # Create all dataset (splits)
    train_dataset = batch_3dmot.utils.dataset.ImageDataset(params=params,
                                                           class_dict=class_dict_used,
                                                           split_name=split_name_train,
                                                           transform=transform)

    val_dataset = batch_3dmot.utils.dataset.ImageDataset(params=params,
                                                         class_dict=class_dict_used,
                                                         split_name=split_name_val,
                                                         transform=transform)

    # Creating data indices for training and validation splits:
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    if params.resnet.shuffle_data:
        np.random.seed(params.resnet.manual_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Define all data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.resnet.batch_size,
                                               sampler=train_sampler,
                                               num_workers=int(params.resnet.workers))

    num_batches_train = len(train_dataset) / int(params.resnet.batch_size)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params.resnet.batch_size,
                                             sampler=val_sampler,
                                             num_workers=int(params.resnet.workers))

    num_batches_val = len(val_dataset) / params.resnet.batch_size

    if test_scene_meta_list is not None:
        test_dataset = batch_3dmot.utils.dataset.ImageDataset(params=params,
                                                              split_name=split_name_test,
                                                              class_dict=class_dict_used,
                                                              transform=transform)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=params.resnet.batch_size,
                                                  num_workers=int(params.resnet.workers))
        num_batches_test = len(test_dataset) / params.resnet.batch_size
    else:
        test_dataset, test_loader, num_batches = None, None, None

else:
    exit('Wrong dataset type')
    train_dataset, val_dataset, test_dataset = None, None, None
    train_loader, val_loader, test_loader = None, None, None

# Create model
resnet_ae = batch_3dmot.models.resnet_fully_conv.ResNetAE().to(device)

if params.resnet.checkpoint != '':
    resnet_ae.load_state_dict(torch.load(params.resnet.checkpoint))

batch_3dmot.utils.dataset.check_mkdir(os.path.join(params.paths.models, 'resnet/'))

model_params = list(resnet_ae.parameters())
optimizer = torch.optim.Adam(model_params, lr=params.resnet.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=params.resnet.scheduler_step_size,
                                            gamma=params.resnet.scheduler_gamma)

# Set up a tensorboard writer
resnet_writer = SummaryWriter(log_dir=os.path.join(params.paths.models,
                                                   'runs/resnet_%s_%s_%s_%s_%s_%s'
                                                   % (str(datetime.datetime.now()).split('.')[0],
                                                      params.main.version,
                                                      params.main.class_dict,
                                                      params.resnet.batch_size,
                                                      params.resnet.lr,
                                                      params.resnet.num_epochs)))


def train(log_interval, model, device, writer, train_loader, val_loader, optimizer, scheduler, save_images: bool):

    for epoch in range(params.resnet.num_epochs):

        train_progress = tqdm(train_loader)
        for i, data in enumerate(train_progress):
            images, labels = data
            # Move data to GPU if possible
            images, labels = images.to(device), labels.to(device).view(-1,)

            optimizer.zero_grad()
            model = model.train()
            reconstructions = model(images)
            loss = loss_function(reconstructions, images) / float(params.resnet.batch_size)

            writer.add_scalar("Loss/train", loss, epoch)
            train_progress.set_description(f"Loss: {loss.item():.4f}")

            """
            if save_images:
                batch_3dmot.utils.dataset.check_mkdir(os.path.join(params.paths.preprocessed_data, 'reconst_img/'))
                torchvision.utils.save_image(images,
                                             params.paths.tmp
                                             + 'reconst_img/train_inputs_'
                                             + str(epoch) + '_'
                                             + str(i) + '.png')

                torchvision.utils.save_image(reconstructions,
                                             params.paths.tmp
                                             + 'reconst_img/train_outputs_'
                                             + str(epoch) + '_'
                                             + str(i) + '.png')
            """

            # show training information in console
            if (i + 1) % log_interval == 0:
                tqdm.write('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batches_train, blue('train'), loss.item()))

            loss.backward()
            optimizer.step()

        writer.flush()

        with torch.no_grad():
            # Run evaluation step

            val_progress = tqdm(val_loader)
            for i, data in enumerate(val_progress):
                images, labels = data
                # Move data to GPU if possible
                images, labels = images.to(device), labels.to(device).view(-1, )

                model = model.eval()
                reconstructions = model(images)

                loss = loss_function(reconstructions, images) / float(params.resnet.batch_size)

                if params.resnet.save_images_val_test:
                    batch_3dmot.utils.dataset.check_mkdir(os.path.join(params.paths.preprocessed_data, 'reconst_img/'))
                    torchvision.utils.save_image(images,
                                                 params.paths.preprocessed_data
                                                 + 'reconst_img/val_inputs'
                                                 + str(epoch) + '_'
                                                 + str(i) + '.png')

                    torchvision.utils.save_image(reconstructions,
                                                 params.paths.preprocessed_data
                                                 + 'reconst_img/val_outputs_'
                                                 + str(epoch) + '_'
                                                 + str(i) + '.png')

                writer.add_scalar("Loss/val", loss, epoch)
                val_progress.set_description(f"Loss: {loss.item():.4f}")

                if (i + 1) % log_interval == 0:
                    tqdm.write('[%d: %d/%d] %s loss: %f ' % (epoch, i, num_batches_val, blue('val'), loss.item()))
            writer.flush()

        scheduler.step()
        torch.save(model.state_dict(), '%s/resnet_epoch%d_%s_%s.pth' % (os.path.join(params.paths.models,'resnet/'),
                                                                        epoch,
                                                                        params.main.version,
                                                                        params.main.class_dict))
    print('Finished Training')
    writer.close()


def inference(model, device, loader, save_images: bool):
    """
    TODO: Define predict method
    Args:
        model:
        device:
        test_loader:

    Returns:

    """

    with torch.no_grad():
        # Run evaluation step

        test_progress = tqdm(loader)
        for i, data in enumerate(test_progress):
            images, labels = data
            # Move data to GPU if possible
            images, labels = images.to(device), labels.to(device).view(-1, )

            model = model.eval()
            encodings, reconstructions = model(images)

            loss = loss_function(reconstructions, images)

            if params.resnet.save_images_val_test:
                torchvision.utils.save_image(images,
                                             params.paths.preprocessed_data
                                             + 'reconst_img/test_inputs'
                                             + str(0) + '_'
                                             + str(i) + '.png')

                torchvision.utils.save_image(reconstructions,
                                             params.paths.preprocessed_data
                                             + 'reconst_img/test_outputs'
                                             + str(0) + '_'
                                             + str(i) + '.png')

            test_progress.set_description(f"Loss: {loss.item():.4f}")

            tqdm.write('[%d/%d] %s loss: %f ' % (i, num_batches_test, blue('test'), loss.item()))


if __name__ == '__main__':
    train(model=resnet_ae, optimizer=optimizer, scheduler=scheduler, device=device, writer=resnet_writer,
          train_loader=train_loader, val_loader=val_loader,
          log_interval=10, save_images=params.resnet.save_images)

    print('Finished Training')
    print('Saving Model...')

