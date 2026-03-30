import torch
import torch.nn as nn
import torch.nn.functional as F

class WAGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_last_layer=False):
        super().__init__()
        # Pesos para o Grafo de Similaridade (Eq. 2 do artigo)
        self.W_1 = nn.Linear(in_dim, in_dim // 2)
        self.W_2 = nn.Linear(in_dim, in_dim // 2)
        
        # Pesos da Convolução em Grafo e Conexão Residual (Eq. 1 do artigo)
        self.W_gcn = nn.Linear(in_dim, out_dim)
        self.conv_res = nn.Conv1d(in_dim, out_dim, 1) 
        
        self.is_last_layer = is_last_layer

    def forward(self, x):
        # x shape: (Batch_Size, T_requests, Feature_Dim)
        B, T, D = x.shape

        # 1. Grafo de Similaridade de Características (A^F)
        xw1 = F.relu(self.W_1(x)) # (B, T, D/2)
        xw2 = F.relu(self.W_2(x)) # (B, T, D/2)
        A_F = torch.softmax(torch.bmm(xw1, xw2.transpose(1, 2)), dim=-1) # (B, T, T)

        # 2. Grafo de Consistência Temporal (A^T)
        # Assumindo que o eixo T já está ordenado cronologicamente (pelo seu dataframe)
        idx = torch.arange(T, device=x.device).float()
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        A_T = torch.exp(-dist).unsqueeze(0).expand(B, T, T) # (B, T, T)

        # Matriz Adjacência Combinada
        A = A_F + A_T

        # 3. Operação GCN
        out = self.W_gcn(torch.bmm(A, x)) # (B, T, out_dim)

        # 4. Conexão Residual (f^i no artigo)
        res = self.conv_res(x.transpose(1, 2)).transpose(1, 2) # (B, T, out_dim)
        
        # Ativação: Sigmoid na última camada, ReLU nas anteriores
        if self.is_last_layer:
            return torch.sigmoid(out + res)
        return F.relu(out + res)