import torch
import torch.nn as nn
import torch.nn.functional as F

class WAGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_last_layer=False):
        super().__init__()
        self.W_1 = nn.Linear(in_dim, in_dim // 2)
        self.W_2 = nn.Linear(in_dim, in_dim // 2)
        
        self.W_gcn = nn.Linear(in_dim, out_dim)
        self.conv_res = nn.Conv1d(in_dim, out_dim, 1) 
        
        self.is_last_layer = is_last_layer

    def forward(self, x, mask=None):
        B, T, D = x.shape

        xw1 = F.relu(self.W_1(x)) 
        xw2 = F.relu(self.W_2(x)) 
        A_F = torch.softmax(torch.bmm(xw1, xw2.transpose(1, 2)), dim=-1)

        idx = torch.arange(T, device=x.device).float()
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        A_T = torch.exp(-dist).unsqueeze(0).expand(B, T, T) 

        A = A_F + A_T
        if mask is not None:
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            A = A.masked_fill(~pair_mask, 0.0)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        out = self.W_gcn(torch.bmm(A, x)) 

        res = self.conv_res(x.transpose(1, 2)).transpose(1, 2) 
        result = out + res
        if mask is not None:
            result = result.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        if self.is_last_layer:
            return torch.sigmoid(result)
        return F.relu(result)
