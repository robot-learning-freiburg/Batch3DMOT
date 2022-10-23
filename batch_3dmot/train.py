import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import math
import argparse
from collections import defaultdict
from tqdm import tqdm 
import numpy as np
import datetime

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils.data
import torch.nn.functional as F
import torch_geometric.data

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification.average_precision import average_precision
from torchmetrics.functional.classification.precision_recall import recall

import batch_3dmot.utils.load_scenes
import batch_3dmot.utils.graph_data
import batch_3dmot.utils.dataset

import batch_3dmot.models.clr_att_gnn

import batch_3dmot.models.pointnet
import batch_3dmot.models.resnet_fully_conv
import batch_3dmot.models.radarnet

from batch_3dmot.utils.config import ParamLib

torch.cuda.empty_cache()


class Batch3DMOT:

    def __init__(self, params: ParamLib,
                 num_epochs: int,
                 batch_size_graph: int,
                 train_scenes: list,
                 val_scenes: list,
                 graph_data_dir: str):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.log_dir = os.path.join(params.paths.models, 'gnn/')

        self.epochs = num_epochs
        self.epoch = 0

        class_dict_used = batch_3dmot.utils.dataset.get_class_config(params=params, class_dict_name=params.main.class_dict)

        self.resnet_ae = batch_3dmot.models.resnet_fully_conv.ResNetAE().to(self.device)
        self.resnet_ae.load_state_dict(torch.load(os.path.join(params.paths.models,params.resnet.checkpoint)))

        self.pointnet_classifier = batch_3dmot.models.pointnet.PointNetClassifier(k=len(class_dict_used), feature_transform=params.pointnet.feature_transform).to(self.device)
        self.pointnet_classifier.load_state_dict(torch.load(os.path.join(params.paths.models, params.pointnet.checkpoint)))

        self.radarnet_classifier = batch_3dmot.models.radarnet.RadarNetClassifier(k=len(class_dict_used), feature_transform=params.radarnet.feature_transform).to(self.device)
        self.radarnet_classifier.load_state_dict(torch.load(os.path.join(params.paths.models, params.radarnet.checkpoint)))

        self.gnn = batch_3dmot.models.clr_att_gnn.GNN(gnn_depth=params.gnn.gnn_depth,
                                                    node_dim=params.gnn.node_dim,
                                                    edge_dim=params.gnn.edge_dim,
                                                    img_encoder=self.resnet_ae,
                                                    lidar_encoder=self.pointnet_classifier,
                                                    radar_encoder=self.radarnet_classifier,
                                                    use_attention=params.gnn.attention,
                                                    )

        # Define model checkpoint to continue training
        chckpt = torch.load(os.path.join(params.paths.models, params.predict.checkpoint), map_location=self.device)
        missing_keys, unexpected_keys = self.gnn.load_state_dict(chckpt, strict=False)
        self.gnn = self.gnn.to(self.device)

        self.train_scenes = train_scenes
        self.val_scenes = val_scenes
        self.batch_size_graph = batch_size_graph
        self.graph_data_dir = graph_data_dir

        wandb.watch(self.gnn)


    def train_dataloader(self):
        train_graphdata = batch_3dmot.utils.graph_data.GraphDataset(params, self.train_scenes, self.graph_data_dir,
                                                                        params.gnn.batch_size_graph, False)
        train_graph_loader = torch_geometric.loader.DataLoader(train_graphdata, batch_size=params.gnn.batch_size,
                                                                num_workers=10, shuffle=True)
        return train_graph_loader

    def val_dataloader(self):
        val_graphdata = batch_3dmot.utils.graph_data.GraphDataset(params, self.val_scenes, self.graph_data_dir, 
                                                                    params.gnn.batch_size_graph, False)
        val_graph_loader = torch_geometric.loader.DataLoader(val_graphdata, batch_size=params.gnn.batch_size,
                                                                num_workers=8, shuffle=True)
        return val_graph_loader

    def train(self):

        torch.cuda.empty_cache()

        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()

        optimizer = torch.optim.Adam(self.gnn.parameters(),
                                        lr=float(params.gnn.lr),  # 3e-4
                                        weight_decay=float(params.gnn.weight_decay),
                                        betas=(params.gnn.beta_lo, params.gnn.beta_hi))

        criterion = torch.nn.BCELoss()

        class_dict_used = batch_3dmot.utils.dataset.get_class_config(params=params, class_dict_name=params.main.class_dict)

        global_train_step = 0
        global_val_step = 0

        for epoch in range(self.epochs):
            self.epoch += 1
            self.gnn.train()

            metrics = defaultdict(list)
            train_progress = tqdm(train_loader)
            for it, data in enumerate(train_progress):
                
                data = data.to(self.device)
                gt = data.y.float()

                gt_embed = gt.clone().detach()
                gt_mask = gt == 0
                gt_embed[gt_mask] = -1

                out, x_sens = self.gnn.forward(data)
                out = out.squeeze(1)

                if params.gnn.loss == "bce":
                    criterion = torch.nn.BCELoss()
                elif params.gnn.loss == "cb":
                    criterion = torch.nn.BCELoss(weight=data.edge_weights)

                edge_loss = criterion(out, gt) / params.gnn.batch_size
                loss = edge_loss
                avg_prec = average_precision(out, gt, pos_label=1)

                for category, cls_idx in class_dict_used.items():
                   
                    if torch.sum(data.edge_classes == cls_idx) > 0:
                        avgprec_logstr = 'avgprec/' + category
                        avgprec_categ = average_precision(out[data.edge_classes == cls_idx], gt[data.edge_classes == cls_idx], pos_label=1) 
                        metrics['train/' + avgprec_logstr].append(avgprec_categ.item())

                metrics['train/loss'].append(loss.item())
                metrics['train/avgprec'].append(avg_prec.item())
                train_progress.set_description(
                    f"Loss: {loss.item():.4f}, AP: {avg_prec.item():.4f}")

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                global_train_step += 1

            with torch.no_grad():

                self.gnn.eval()
                val_progress = tqdm(val_loader)

                for i, data_val in enumerate(val_progress):

                    data_val = data_val.to(self.device)
                    gt_val = data_val.y.float()

                    out_val, x_sens_val = self.gnn.forward(data_val)
                    out_val = out_val.squeeze(1)

                    gt_embed_val = gt_val.clone().detach()
                    gt_mask = gt_val == 0
                    gt_embed_val[gt_mask] = -1

                    # Edge loss
                    if params.gnn.loss == "bce":
                        criterion = torch.nn.BCELoss()
                    elif params.gnn.loss == "cb":
                        criterion = torch.nn.BCELoss(weight=data_val.edge_weights)


                    edge_loss = criterion(out_val, gt_val) / params.gnn.batch_size
                    loss = edge_loss
                    
                    avg_prec = average_precision(out_val, gt_val, pos_label=1)

                    for category, cls_idx in class_dict_used.items():
                    
                        if torch.sum(data_val.edge_classes == cls_idx) > 0:
                            avgprec_logstr = 'avgprec/' + category
                            avgprec_categ = average_precision(out_val[data_val.edge_classes == cls_idx], 
                                                                gt_val[data_val.edge_classes == cls_idx], 
                                                                pos_label=1) 

                            metrics['val/' + avgprec_logstr].append(avgprec_categ.item())

                    metrics['val/loss'].append(loss.item())
                    metrics['val/avgprec'].append(average_precision(out_val, gt_val, pos_label=1).cpu().numpy())
                    val_progress.set_description(
                    f"Val Ep {epoch}: Loss: {loss.item():.4f}, AP: {avg_prec.item():.4f}")

                    global_val_step += 1

            metrics = {k: np.nanmean(v) for k, v in metrics.items()}

            tqdm.write('train/avgprec:' + str(metrics['train/avgprec']))
            tqdm.write('val/avgprec:' + str(metrics['val/avgprec']))
            tqdm.write('AP/bicycle: ' + str(metrics['val/avgprec/bicycle']) + ' AP/bus:' + str(metrics['val/avgprec/bus']) + ' AP/car:' + str(metrics['val/avgprec/car']) + 'AP/motorcycle: ' + str(metrics['val/avgprec/motorcycle'])
                         + ' AP/pedestrian:' + str(metrics['val/avgprec/pedestrian']) + ' AP/trailer:' + str(metrics['val/avgprec/trailer']) + ' AP/truck:' + str(metrics['val/avgprec/truck']))

            if epoch > -1:

                checkpoint_path = '%sgnn/gnn_epoch%d_%s__TrainAP%s_ValAP%s.pth' % (
                                    self.log_dir,
                                    epoch,
                                    params.main.version,
                                    str(round(metrics['train/avgprec'], 6)),
                                    str(round(metrics['val/avgprec'], 6)))

                torch.save(self.gnn.state_dict(), checkpoint_path)


if __name__ == '__main__':
    # ----------- PARAMETER SOURCING --------------

    parser = argparse.ArgumentParser()

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Specific arguments (namespace: graph_construction)
    parser.add_argument('--batch_size_graph', type=int, help='define number of frames contained in one graph')
    parser.add_argument('--checkpoint', type=str, help='model path')
    parser.add_argument('--lr', type=str, help='model path')
    parser.add_argument('--epochs', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.gnn.overwrite(opt)
    
    # Load lists with all scene meta data based on the dataset, the version,
    nusc, meta_lists = batch_3dmot.utils.load_scenes.load_scene_meta_list(data_path=params.paths.data,
                                                                          dataset=params.main.dataset,
                                                                          version=params.main.version)

    train_scene_meta_list, val_scene_meta_list = meta_lists
    graph_data_dir = params.paths.graphs_clr_centerpoint_1440_flip_40nn_iou_14feb_len2

    batch_3dmot_gnn = Batch3DMOT(params=params,
                                 num_epochs=params.gnn.num_epochs,
                                 train_scenes=train_scene_meta_list,
                                 val_scenes=val_scene_meta_list,
                                 graph_data_dir=graph_data_dir,
                                 batch_size_graph=params.gnn.batch_size_graph)

    batch_3dmot_gnn.train()

