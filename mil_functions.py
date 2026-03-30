import torch
from torch.functional import F

def mil_loss(scores, labels, k=3):
    """
    scores: (Batch, T_requests, 1)
    labels: (Batch,) -> 1 para sessão com Bot, 0 para sessão Normal
    """
    B = scores.shape[0]
    loss = 0.0
    for i in range(B):
        # Pega os k maiores scores da sessão
        topk_scores, _ = torch.topk(scores[i, :, 0], k)
        mean_score = torch.mean(topk_scores)
        
        if labels[i] == 1: # Bot
            loss += (1 - mean_score) ** 2  # Queremos score perto de 1
        else: # Normal
            loss += mean_score ** 2        # Queremos score perto de 0
            
    return loss / B

def contrastive_clustering_loss(features, scores, labels, memory_bank, k=3, tau=0.1):
    """
    Implementa a intuição de puxar para o cluster certo e empurrar do errado.
    """
    B = features.shape[0]
    c_n = memory_bank.normal_center.to(features.device)
    c_a = memory_bank.abnormal_center.to(features.device)
    
    loss = 0.0
    for i in range(B):
        # Seleciona as features das k requisições mais suspeitas (ou normais)
        topk_idx = torch.topk(scores[i, :, 0], k)[1]
        bag_features = features[i, topk_idx, :].mean(dim=0) # Média das top-k (conforme artigo)

        if labels[i] == 1:
            pos_center, neg_center = c_a, c_n
        else:
            pos_center, neg_center = c_n, c_a

        # Similaridade de Coseno (Dual InfoNCE)
        sim_pos = torch.exp(F.cosine_similarity(bag_features.unsqueeze(0), pos_center.unsqueeze(0)) / tau)
        sim_neg = torch.exp(F.cosine_similarity(bag_features.unsqueeze(0), neg_center.unsqueeze(0)) / tau)
        
        # Loss InfoNCE padrão
        loss += -torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))

    return loss / B