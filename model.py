import torch.nn as nn
import os
from modules import resnet18
import torch

class gaze_network(nn.Module):
    def __init__(self, image_net_model):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet18(pretrained=False)
        self.gaze_network.load_state_dict(torch.load(os.path.join(image_net_model, 'resnet18-5c106cde.pth')), strict=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(512, 2),
        )


    def forward(self, x):
        feature_list = self.gaze_network(x)
        out_feature = feature_list[-1]
        out_feature = out_feature.view(out_feature.size(0), -1)
        gaze = self.gaze_fc(out_feature)

        return gaze