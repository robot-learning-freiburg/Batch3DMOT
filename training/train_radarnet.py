#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2022
@author: Martin Buechner, buechner@cs.uni-freiburg.de
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import random
from tqdm import tqdm
import numpy as np
import datetime

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.tensorboard import SummaryWriter

import batch_3dmot.utils.dataset
from batch_3dmot.utils.config import ParamLib
from batch_3dmot.utils.load_scenes import load_scene_meta_list

import batch_3dmot.models.radarnet

# ----------- PARAMETER SOURCING --------------

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--version', type=str, help="define the dataset version that is used")

# Specific arguments
parser.add_argument('--batch_size', type=int, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers')
parser.add_argument('--num_epochs', type=int,  help='number of epochs to train for')
parser.add_argument('--feature_transform', type=bool, help="use feature transform")
parser.add_argument('--shuffle_data', type=bool, help="define whether to shuffle the provided data")

opt = parser.parse_args()

params = ParamLib(opt.config)
params.main.overwrite(opt)
params.radarnet.overwrite(opt)

# Use this lambda function to convert variable names to strings for print statements (in blue color)
blue = lambda x: '\033[94m' + x + '\033[0m'

# Define the used seed to ensure reproducibility
print("Used Seed:", params.radarnet.manual_seed)
random.seed(params.radarnet.manual_seed)
torch.manual_seed(params.radarnet.manual_seed)

# Detect CUDA resources
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
print("Using", torch.cuda.device_count(), "GPU!")

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])

if params.main.dataset == 'nuscenes':
    # Load lists with all scene meta data based on the dataset, the version,
    nusc, meta_lists = load_scene_meta_list(data_path=params.paths.data,
                                            dataset=params.main.dataset,
                                            version=params.main.version)

    if params.main.version == "v1.0-mini":
        train_scene_meta_list, val_scene_meta_list = meta_lists
        test_scene_meta_list = None

        train_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, train_scene_meta_list)
        val_tokens =  batch_3dmot.utils.dataset.create_all_split_tokens(nusc, val_scene_meta_list)
        test_tokens = None
        split_name_train = 'mini_train'
        split_name_val = 'mini_val'
        split_name_test = None

    elif params.main.version == "v1.0-trainval":
        train_scene_meta_list, val_scene_meta_list = meta_lists
        test_scene_meta_list = None

        train_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, train_scene_meta_list)
        val_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, val_scene_meta_list)
        split_name_train = 'train'
        split_name_val = 'val'
        split_name_test = 'test'

    elif params.main.version == "v1.0-test":
        train_scene_meta_list = None
        val_scene_meta_list = None
        test_scene_meta_list = meta_lists

        train_tokens, val_tokens = None, None
        test_tokens = batch_3dmot.utils.dataset.create_all_split_tokens(nusc, test_scene_meta_list)
        split_name_train = None
        split_name_val = None
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
    train_dataset = batch_3dmot.utils.dataset.RadarDataset(params=params,
                                                           split_name=split_name_train,
                                                           class_dict=class_dict_used)

    val_dataset = batch_3dmot.utils.dataset.RadarDataset(params=params,
                                                         split_name=split_name_val,
                                                         class_dict=class_dict_used)

    # Creating data indices for training and validation splits:
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))

    if params.radarnet.shuffle_data:
        np.random.seed(params.radarnet.manual_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Define all data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.radarnet.batch_size,
                                               sampler=train_sampler,
                                               num_workers=int(params.radarnet.workers),
                                               collate_fn=batch_3dmot.utils.dataset.collate_radar)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params.radarnet.batch_size,
                                             sampler=val_sampler,
                                             num_workers=int(params.radarnet.workers),
                                             collate_fn=batch_3dmot.utils.dataset.collate_radar)

    if test_scene_meta_list is not None:
        test_dataset = batch_3dmot.utils.dataset.RadarDataset(params=params,
                                                              split_name=split_name_test,
                                                              class_dict=class_dict_used)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=params.radarnet.batch_size,
                                                  num_workers=int(params.radarnet.workers),
                                                  collate_fn=batch_3dmot.utils.dataset.collate_radar)
    else:
        test_dataset, test_loader = None, None

else:
    exit('Wrong dataset type')
    train_dataset, val_dataset, test_dataset = None, None, None
    train_loader, val_loader, test_loader = None, None, None
    num_classes = None

try:
    if not os.path.exists(os.path.join(params.paths.models, "radarnet/")):
        os.mkdir(os.path.join(params.paths.models, "radarnet/"))
except OSError:
    pass


classifier = batch_3dmot.models.radarnet.RadarNetClassifier(k=num_classes, feature_transform=params.radarnet.feature_transform)

if params.radarnet.checkpoint != '':
    classifier.load_state_dict(torch.load(params.radarnet.checkpoint))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Move model to GPU (if CUDA available)
classifier.to(device)

# set up tensorboard writer
radarnet_writer = SummaryWriter(log_dir=os.path.join(params.paths.models,
                                                   'runs/radarnet_%s_%s_%s_%s_%s_%s'
                                                   % (str(datetime.datetime.now()).split('.')[0],
                                                      params.main.version,
                                                      params.main.class_dict,
                                                      params.radarnet.batch_size,
                                                      params.radarnet.lr,
                                                      params.radarnet.num_epochs)))

num_batches_train = len(train_dataset) / params.radarnet.batch_size
num_batches_val = len(val_dataset) / params.radarnet.batch_size


def train(model, device, writer, train_loader, val_loader):
    """
    TODO: Define combined train/val method for the radarnet training
    Args:
        model:
        device:
        writer:
        train_loader:
        val_loader:

    Returns:

    """

    for epoch in range(params.radarnet.num_epochs):

        train_progress = tqdm(train_loader)
        for i, data in enumerate(train_progress):
            radarpoints, targets = data
            # Move data to GPU if possible
            radarpoints, targets = radarpoints.to(device), targets.to(device)

            optimizer.zero_grad()
            model = model.train()
            pred, feat = model(radarpoints.float())
            loss = torch.nn.functional.nll_loss(pred, targets) / float(params.radarnet.batch_size)

            writer.add_scalar("Loss/train", loss, epoch)
            train_progress.set_description(f"Loss: {loss.item():.4f}")

            '''
            if params.radarnet.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            '''

            loss.backward()
            optimizer.step()

            # Choose class with highest softmax probability
            pred_choice = pred.data.max(1)[1]

            # Compute accuracy, then log and print
            correct = pred_choice.eq(targets.data).cpu().sum()
            acc = correct.item() / float(params.radarnet.batch_size)

            writer.add_scalar("Acc/train", acc, epoch)
            tqdm.write('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batches_train, loss.item(), acc))
            writer.flush()

        with torch.no_grad():
            # Run evaluation step

            val_progress = tqdm(val_loader)
            for i, data in enumerate(val_progress):
                pointclouds, targets = data
                pointclouds = pointclouds.to(device)
                targets = targets.to(device)

                model = model.eval()
                pred, _ = model(pointclouds.float())

                loss = torch.nn.functional.nll_loss(pred, targets) / float(params.radarnet.batch_size)

                writer.add_scalar("Loss/val", loss, epoch)
                val_progress.set_description(f"Loss: {loss.item():.4f}")

                # Choose class with highest softmax probability
                pred_choice = pred.data.max(1)[1]

                # Compute accuracy, then log and print
                correct = pred_choice.eq(targets.data).cpu().sum()
                acc = correct.item() / float(params.radarnet.batch_size)

                writer.add_scalar("Acc/val", acc, epoch)
                tqdm.write('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batches_val, blue('val'), loss.item(), acc))

        scheduler.step()
        torch.save(classifier.state_dict(),
                   '%s/radarnet_epoch%d_%s_%s.pth' % (os.path.join(params.paths.models,'radarnet/'),
                                               epoch,
                                               params.main.version,
                                               params.main.class_dict))



def inference(model, device, test_loader):
    """
    TODO: Define predict method
    Args:
        model:
        device:
        test_loader:

    Returns:

    """
    total_correct = 0
    total_testset = 0

    for i, data in tqdm(enumerate(test_loader, 0)):
        data = data.to(device)
        points, target = data

        model = model.eval()
        pred, _, _ = model(points.float())

        print("############ EPOCH " + str(i) + ":")
        print(pred)
        print(target)
        print("---------------------------------")

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__ == '__main__':
    train(classifier, device, radarnet_writer, train_loader, val_loader)