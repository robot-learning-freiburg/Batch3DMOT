import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(torch.nn.Module):

    """
    Input transform using a 'T-Net'. The conv1d layers resemble MLPs.
    """

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9)
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)


    def forward(self, x):
        """
        Run forward pass through input transform T-Net.
        Args:
            x: unordered raw pointcloud (n x 3)
        Returns
            x: transformed point cloud with dims nx3
        """
        batchsize = x.size()[0]
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = torch.nn.functional.relu(self.bn4(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix to transformed x
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(torch.nn.Module):
    """
    Feature transform T-Net with output 64x64. Matrix initialized as identity matrix.
    Uses regularization loss (weight 0.001) at softmax loss to induce orthogonality.
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k*k)
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        """
        Run forward pass through input transform T-Net.
        Args:
            x: unordered raw pointcloud (n x 3)
        Returns
            x: transformed point cloud with dims nx3
        """
        batchsize = x.size()[0]
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = torch.nn.functional.relu(self.bn4(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix to transformed x
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(torch.nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetFeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):

        n_pts = x.size()[2]

        # Pass through input transform T-Net (3 x 3)-transform
        trans = self.stn(x)
        x = x.transpose(2, 1)

        # T-Net matrix multiplication
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # Run through first shared mlp: (64,64)-mlp=conv1
        x = F.relu(self.bn1(self.conv1(x)))

        # Pass through feature transform T-Net (64 x 64)-transform
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            # T-Net 64x64 matrix multiplication
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        # Perform forward pass through mlp (4,128,1024)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Either return (1024-global feature) or (pointfeat + 1024-global feature)
        if self.global_feat:
            # For classification
            return x, trans, trans_feat
        else:
            # For segmentation
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetClassifier(torch.nn.Module):
    def __init__(self, k=7, feature_transform=False):
        super(PointNetClassifier, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1), trans, trans_feat

    def forward_feat(self, x):
        x, trans, trans_feat = self.feat(x)
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.dropout(self.fc2(x))))
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


class PointNetDecoder(torch.nn.Module):
    """
    Use a FC decoder to reconstruct the point cloud.
    Requires the PyTorch3D Chamfer distance loss (not yet tried out).
    """
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetDecoder, self).__init__()

        self.fc1 = torch.nn.Linear(1024, 1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(1024, 1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)

        self.fc3 = torch.nn.Linear(1024, 3)
        self.bn3 = torch.nn.BatchNorm1d(3)

    def forward(self, num_points, global_feat):

        out = self.bn1(self.fc1(global_feat))
        out = self.bn2(self.fc2(out))
        out = self.fc3(out)
        out = out.view(-1, num_points*3)

        return out