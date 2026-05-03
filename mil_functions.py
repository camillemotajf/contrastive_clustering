import torch
import torch.nn.functional as F

def mil_loss(scores, labels, k=3, mask=None):
  
    B = scores.shape[0]
    loss = 0.0
    for i in range(B):
        valid_scores = scores[i, :, 0] if mask is None else scores[i, mask[i], 0]
        topk_scores, _ = torch.topk(valid_scores, min(k, valid_scores.shape[0]))
        mean_score = torch.mean(topk_scores)
        
        if labels[i] == 1: # Bot
            loss += (1 - mean_score) ** 2 
        else: # Normal
            loss += mean_score ** 2       
            
    return loss / B

def contrastive_clustering_loss(features, scores, labels, memory_bank, k=3, tau=0.1, mask=None):
  
    B = features.shape[0]
    c_n = memory_bank.normal_center.to(features.device)
    c_a = memory_bank.abnormal_center.to(features.device)
    
    loss = 0.0
    for i in range(B):
        valid_scores = scores[i, :, 0] if mask is None else scores[i, mask[i], 0]
        valid_features = features[i] if mask is None else features[i, mask[i]]
        topk_idx = torch.topk(valid_scores, min(k, valid_scores.shape[0]))[1]
        bag_features = valid_features[topk_idx, :].mean(dim=0)

        if labels[i] == 1:
            pos_center, neg_center = c_a, c_n
        else:
            pos_center, neg_center = c_n, c_a

        sim_pos = torch.exp(F.cosine_similarity(bag_features.unsqueeze(0), pos_center.unsqueeze(0)) / tau)
        sim_neg = torch.exp(F.cosine_similarity(bag_features.unsqueeze(0), neg_center.unsqueeze(0)) / tau)
        
        loss += -torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))

    return loss / B
