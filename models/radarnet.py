import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class RadarNetFeat(torch.nn.Module):
    def __init__(self):
        super(RadarNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):

        num_pts = x.size()[2]
        #print(x.shape)
        # Run through first shared mlp: (64,64)-mlp=conv1
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        # Perform forward pass through mlp (4,128,1024)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        x = self.bn3(self.conv3(x))
        #print(x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        #print(x.shape)
        feat = x.view(-1, 1024)
        #print(feat.shape)

        # Return (1024-global feature) for classification
        return x, feat


class RadarNetClassifier(torch.nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(RadarNetClassifier, self).__init__()
        self.feat = RadarNetFeat()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x, feat = self.feat(x)
        x = self.fc1(feat)
        x = torch.nn.functional.relu(self.bn1(x))
        x = torch.nn.functional.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1), feat

    def forward_feat(self, x):
        x, feat = self.feat(x)
        x = torch.nn.functional.relu(self.bn1(self.fc1(feat)))
        x = torch.nn.functional.relu(self.bn2(self.dropout(self.fc2(x))))
        return x


class RadarNetDecoder(torch.nn.Module):
    """
    Use a FC decoder to reconstruct the point cloud.
    Requires the PyTorch3D Chamfer distance loss (not yet tried out).
    """
    def __init__(self, global_feat=True, feature_transform=False):
        super(RadarNetDecoder, self).__init__()

        self.fc1 = torch.nn.Linear(1024, 1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(1024, 1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)

        self.fc3 = torch.nn.Linear(1024, 4)
        self.bn3 = torch.nn.BatchNorm1d(4)

    def forward(self, num_points, global_feat):

        out = self.bn1(self.fc1(global_feat))
        out = self.bn2(self.fc2(out))
        out = self.fc3(out)
        out = out.view(-1, num_points*3)

        return out