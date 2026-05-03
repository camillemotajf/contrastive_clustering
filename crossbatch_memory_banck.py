import torch
from torch import nn
from wagcn_layer import WAGCNLayer

class CrossBatchMemoryBank:
    def __init__(self, feature_dim):
        self.normal_center = torch.zeros(feature_dim)
        self.abnormal_center = torch.zeros(feature_dim)

    def update(self, normal_features, abnormal_features, momentum=0.9):
        if normal_features.numel() > 0:
            self.normal_center = momentum * self.normal_center + (1 - momentum) * normal_features.mean(dim=0)
        if abnormal_features.numel() > 0:
            self.abnormal_center = momentum * self.abnormal_center + (1 - momentum) * abnormal_features.mean(dim=0)
        
        self.normal_center = self.normal_center.detach()
        self.abnormal_center = self.abnormal_center.detach()

class BotDetectionNet(nn.Module):
    def __init__(self, feature_dim, embedding_dim=32):
        super().__init__()
        self.layer1 = WAGCNLayer(feature_dim, 128)
        self.layer2 = WAGCNLayer(128, 64)
        self.layer3 = WAGCNLayer(64, embedding_dim)
        
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x, mask=None):
        h1 = self.layer1(x, mask=mask) 
        h2 = self.layer2(h1, mask=mask)
        h3 = self.layer3(h2, mask=mask)
        
        scores = torch.sigmoid(self.classifier(h3)) 
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), 0.0)
        return scores, h3
