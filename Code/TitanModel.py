# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:38:03 2020

@author: RML
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TitanModel(nn.Module):
    def __init__(self):
        super(TitanModel, self).__init__()
        self.fc1 = nn.Linear(10, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 30)
        self.bn3 = nn.BatchNorm1d(30)
        self.fc4 = nn.Linear(30, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        
        x = F.sigmoid(x)
        return x

def _titanmodel(pretrained=False, path=None):
    model = TitanModel()
    if pretrained:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    return model
