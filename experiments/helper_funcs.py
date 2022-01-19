import numpy as np
import torch
from torch import nn

def class_card(prev_labels,
               x,
               num_classes,
               DMGT_model,
               rare_isoreg,
               common_isoreg,
               device):
    
    DMGT_model.eval()
    logits = DMGT_model(x.unsqueeze(0).to(device))
    softmax = nn.functional.softmax(logits, dim=1)
    
    top_score = softmax.max(1).values
    pred = softmax.max(1).indices
    
    if pred < 5:
        cal_top_score = rare_isoreg.predict(top_score.cpu().detach().numpy())
        
    else:
        cal_top_score = common_isoreg.predict(top_score.cpu().detach().numpy())

    renorm_factor = (1 - torch.from_numpy(cal_top_score).to(device))/(softmax.sum() - top_score) if top_score < 1 else 0
    softmax = torch.mul(renorm_factor*torch.ones(num_classes).to(device), softmax)
    softmax.squeeze().double()[pred] = torch.from_numpy(cal_top_score).to(device)
    
    softmax = softmax.squeeze()
    label_counts = [(prev_labels==i).float().sum() for i in range(num_classes)]
    
    return sum([softmax[i] * (np.sqrt(label_counts[i] + 1) - np.sqrt(label_counts[i])) for i in range(num_classes)]) 

def get_subsets(stream_x,
                stream_y,
                cost, 
                DMGT_model,
                num_classes,
                rare_isoreg,
                common_isoreg,
                device):
    
    DMGT_x = stream_x[0].unsqueeze(0)
    DMGT_y = stream_y[0].unsqueeze(0)
    
    DMGT_model.eval()
    
    for i in range(1, len(stream_x)):

        if class_card(DMGT_y,
                      stream_x[i],
                      num_classes,
                      DMGT_model,
                      rare_isoreg,
                      common_isoreg,
                      device) > cost:
        
            DMGT_x = torch.cat((DMGT_x, stream_x[i].unsqueeze(0)))
            DMGT_y = torch.cat((DMGT_y, stream_y[i].unsqueeze(0)))
    
    rand_idxs = torch.randperm(len(stream_x))[:len(DMGT_x)]
    RAND_x = stream_x[rand_idxs]
    RAND_y = stream_y[rand_idxs]
    
    return DMGT_x, DMGT_y, RAND_x, RAND_y
