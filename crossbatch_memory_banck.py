import torch
from wagcn_layer import WAGCNLayer

class CrossBatchMemoryBank:
    def __init__(self, feature_dim):
        # Inicializa os centros de cluster
        self.normal_center = torch.zeros(feature_dim)
        self.abnormal_center = torch.zeros(feature_dim)

    def update(self, normal_features, abnormal_features, momentum=0.9):
        # Atualização via média móvel exponencial (EMA) para estabilidade
        if normal_features.numel() > 0:
            self.normal_center = momentum * self.normal_center + (1 - momentum) * normal_features.mean(dim=0)
        if abnormal_features.numel() > 0:
            self.abnormal_center = momentum * self.abnormal_center + (1 - momentum) * abnormal_features.mean(dim=0)
        
        self.normal_center = self.normal_center.detach()
        self.abnormal_center = self.abnormal_center.detach()

class BotDetectionNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Estrutura de 3 camadas WAGCN
        self.layer1 = WAGCNLayer(feature_dim, 128)
        self.layer2 = WAGCNLayer(128, 64)
        self.layer3 = WAGCNLayer(64, 32)
        
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        h1 = self.layer1(x) # Usado para o clustering
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        
        scores = torch.sigmoid(self.classifier(h3)) # (B, T, 1)
        return scores, h1