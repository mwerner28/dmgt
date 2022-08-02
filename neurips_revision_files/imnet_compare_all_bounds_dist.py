import torch 
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split, sampler
import torchvision 
from torchvision.models import resnet50, resnet18, wide_resnet50_2
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
import numpy as np
import random
from tqdm.autonotebook import tqdm
import inspect
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import pdb
import tracemalloc
import gc 
import shutil
#import line_profiler
import atexit
import os
from os.path import exists as file_exists
from tqdm import tqdm
import copy
import seaborn as sns
import pandas as pd
import sympy
from sympy.solvers import solve 
from sympy import Symbol 
import argparse
import torch.multiprocessing
from numpy import genfromtxt
from datetime import datetime
from PIL import Image
from sklearn.isotonic import IsotonicRegression
torch.multiprocessing.set_sharing_strategy('file_system')
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

now = datetime.now()    
date_time = now.strftime('%m.%d_%H.%M.%S')
print(date_time)

def class_card(prev_labels,
               x,
               num_classes,
               model,
               is_isoreg,
               rare_isoreg,
               common_isoreg,
               device):
    
    model.eval()
    logits = model(x.unsqueeze(0).to(device))
    softmax = nn.functional.softmax(logits, dim=1)
    
    top_score = softmax.max(1).values
    pred = softmax.max(1).indices
    
    if is_isoreg:
        if pred < 5:
            cal_top_score = rare_isoreg.predict(top_score.cpu().detach().numpy())
            
        else:
            cal_top_score = common_isoreg.predict(top_score.cpu().detach().numpy())
    
        renorm_factor = (1 - torch.from_numpy(cal_top_score).to(device))/(softmax.sum() - top_score) if top_score < 1 else 0
        softmax = torch.mul(renorm_factor*torch.ones(num_classes).to(device), softmax)
        softmax.squeeze().double()[pred] = torch.from_numpy(cal_top_score).to(device)
    
    softmax = softmax.squeeze()
    label_counts = [(prev_labels==i).float().sum() for i in range(num_classes)]
    
    return sum([softmax[i] * np.sqrt(label_counts[i] + 1) for i in range(num_classes)]), sum([softmax[i] * np.sqrt(label_counts[i]) for i in range(num_classes)])

def get_DIST_DMGT_subsets(stream_x,
                          stream_y,
                          taus,
                          sel_round,
                          DIST_DMGT_model,
                          num_classes,
                          is_isoreg,
                          rare_isoreg,
                          common_isoreg,
                          device,
                          budget):
    print('tau', sel_round, taus[sel_round])

    print('getting DIST DMGT subsets') 
    DIST_DMGT_x = stream_x[0].unsqueeze(0)
    DIST_DMGT_y = stream_y[0].unsqueeze(0)
    
    for i in range(1, len(stream_x)):
        f_values =  class_card(DIST_DMGT_y,
                      stream_x[i],
                      num_classes,
                      DIST_DMGT_model,
                      is_isoreg,
                      rare_isoreg,
                      common_isoreg,
                      device)
        
        if f_values[0] - f_values[1] >= taus[int(sel_round)] and len(DIST_DMGT_x) < budget:
            DIST_DMGT_x = torch.cat((DIST_DMGT_x, stream_x[i].unsqueeze(0)))
            DIST_DMGT_y = torch.cat((DIST_DMGT_y, stream_y[i].unsqueeze(0)))
    
    rand_idxs = torch.randperm(len(stream_x))[:budget]
    RAND_x = stream_x[rand_idxs]
    RAND_y = stream_y[rand_idxs]
    return DIST_DMGT_x, DIST_DMGT_y, RAND_x, RAND_y

def get_SEIVE_subsets(stream_x,
                      stream_y,
                      SEIVE_model,
                      num_classes,
                      is_isoreg,
                      rare_isoreg,
                      common_isoreg,
                      device,
                      budget,
                      epsilon):

    print('getting seive subsets')
    SEIVE_x = stream_x[0].unsqueeze(0)
    SEIVE_y = stream_y[0].unsqueeze(0)
    
    init_x = SEIVE_x[0]
    init_y = SEIVE_y[0]

    m = 1
    epsilon = 0.1
    j = 1
    O = []
    while (1+epsilon)**j >= m and (1+epsilon)**j <= 2*m*len(stream_x):
        O += [(1+epsilon)**j]
        j += 1
    set_dict = {}
    taus = torch.Tensor().to(device)
    for i in range(1, len(stream_x)):
        new_taus = torch.Tensor().to(device)
        for idx,v in enumerate(O):
            if v not in list(set_dict.keys()):
                set_dict[v] = [(init_x,init_y)]
                taus = torch.cat((taus,torch.tensor([[v]]).to(device)))
            else:
                f_values = class_card(torch.tensor(list(zip(*set_dict[v]))[1]),
                                      stream_x[i],
                                      num_classes,
                                      SEIVE_model,
                                      is_isoreg,
                                      rare_isoreg,
                                      common_isoreg,
                                      device)
                tau = (v/2 - f_values[1])/(budget - len(set_dict[v]))
                new_taus = torch.cat((new_taus,torch.tensor([tau]).to(device)))
                if f_values[0] - f_values[1] >= tau and len(set_dict[v]) < budget:
                    set_dict[v] += [(stream_x[i],stream_y[i])]
        if len(new_taus) > 0:
            taus = torch.cat((taus, new_taus.unsqueeze(1)),dim=1) 

    max_value = -np.inf
    max_key = None
    max_idx = 0
    for idx,key in enumerate(list(set_dict.keys())):
        label_counts = [(torch.tensor(list(zip(*set_dict[key]))[1])==i).float().sum() for i in range(num_classes)]
        if sum([np.sqrt(label_counts[i]) for i in range(num_classes)]) > max_value:
            max_key = key
            max_idx = idx
            max_value = sum([np.sqrt(label_counts[i]) for i in range(num_classes)])
    SEIVE_x = torch.stack(list(zip(*set_dict[max_key]))[0])
    SEIVE_y = torch.stack(list(zip(*set_dict[max_key]))[1])
    sel_taus = taus[max_idx][1:]
    SEIVE_min_max_taus = torch.tensor([sel_taus.min(),sel_taus.max()])
    
    return SEIVE_x, SEIVE_y, SEIVE_min_max_taus

def get_SEIVE_PLUS_subsets(stream_x,
                           stream_y,
                           SEIVE_PLUS_model,
                           num_classes,
                           is_isoreg,
                           rare_isoreg,
                           common_isoreg,
                           device,
                           budget,
                           epsilon):

    print('getting seive++ subsets')
    SEIVE_PLUS_x = stream_x[0].unsqueeze(0)
    SEIVE_PLUS_y = stream_y[0].unsqueeze(0)
    
    init_x = SEIVE_PLUS_x[0]
    init_y = SEIVE_PLUS_y[0]

    m = 1
    epsilon = 0.1
    lb = 0
    set_dict = {}
    taus = torch.Tensor().to(device)
    for i in range(1, len(stream_x)):
        print('i',i)
        new_taus = torch.Tensor().to(device)
        tau_min = max(lb,m)/(2*budget)

        j = 1
        O_i = []
        while (1+epsilon)**j >= tau_min/(1+epsilon) and (1+epsilon)**j <= 1:
            O_i += [(1+epsilon)**j]
            j += 1
        
        set_dict = {tau:set_dict[tau] for tau in set_dict if tau > tau_min}
        for idx,tau in enumerate(O_i):
            if tau not in list(set_dict.keys()):
                set_dict[tau] = [(init_x,init_y)]
                taus = torch.cat((taus,torch.tensor([[tau]]).to(device)))
            else:
                f_values = class_card(torch.tensor(list(zip(*set_dict[tau]))[1]),
                                      stream_x[i],
                                      num_classes,
                                      SEIVE_model,
                                      is_isoreg,
                                      rare_isoreg,
                                      common_isoreg,
                                      device)
                new_taus = torch.cat((new_taus,torch.tensor([tau]).to(device)))
                if tau > 0 and f_values[0] - f_values[1] >= tau and len(set_dict[tau]) < budget:
                    set_dict[tau] += [(stream_x[i],stream_y[i])]
                    lb = max(lb, sum([np.sqrt(label_counts[i] for i in range(num_classes))]))
        if len(new_taus) > 0:
            taus = torch.cat((taus, new_taus.unsqueeze(1)),dim=1) 

    max_value = -np.inf
    max_key = None
    max_idx = 0
    for idx,key in enumerate(list(set_dict.keys())):
        label_counts = [(torch.tensor(list(zip(*set_dict[key]))[1])==i).float().sum() for i in range(num_classes)]
        if sum([np.sqrt(label_counts[i]) for i in range(num_classes)]) > max_value:
            max_key = key
            max_idx = idx
            max_value = sum([np.sqrt(label_counts[i]) for i in range(num_classes)])
    SEIVE_PLUS_x = torch.stack(list(zip(*set_dict[max_key]))[0])
    SEIVE_PLUS_y = torch.stack(list(zip(*set_dict[max_key]))[1])
    sel_taus = taus[max_idx][1:]
    SEIVE_PLUS_min_max_taus = torch.tensor([sel_taus.min(),sel_taus.max()])
    
    return SEIVE_PLUS_x, SEIVE_PLUS_y, SEIVE_PLUS_min_max_taus

def train(device,
          num_epochs,
          train_loader,
          class_dict,
          model):
   
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        
        train_loss = 0.0
        train_acc = 0.0
        
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (output.max(1)[1]==targets).sum().item()
        
        train_loss = train_loss/len(train_loader.dataset)
        train_acc = train_acc/len(train_loader.dataset)
        
        print(f'Epoch: {epoch} Train Loss: {train_loss:.6f}  Train Acc: {train_acc:.6f}')

        if train_acc >= 0.99:
            break

    return model

def load_model(model,
               embed_dim,
               num_classes,
               device):
    
    model_copy = LogRegModel(embed_dim, num_classes).to(device)
    model_copy.load_state_dict(model.state_dict())
    model_copy = model_copy.to(device)
    
    return model_copy

def calc_acc(model,
             test_loader,
             num_classes):

    model.eval()
    
    all_acc = []
    rare_acc = []
    
    for test_idx, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)
        
        rare_idxs = (test_y < num_classes/2).nonzero()
        with torch.no_grad():
            
            batch_preds = torch.argmax(model(test_x), dim=1)
            all_acc += [batch_preds.eq(test_y),]
            rare_acc += [batch_preds.eq(test_y)[rare_idxs],]
    
    all_acc = torch.cat(all_acc, dim=0)
    rare_acc = torch.cat(rare_acc, dim=0)

    return rare_acc.float().mean().unsqueeze(0), all_acc.float().mean().unsqueeze(0)

def calc_cal_acc(is_isoreg,
                 rare_isoreg,
                 common_isoreg,
                 val_loader,
                 model,
                 num_sm_bins,
                 num_classes,
                 device):

    model.eval()
    score_bins = np.arange(num_sm_bins)/num_classes
    
    rare_scores_dict = {i: [] for i in score_bins}
    common_scores_dict = {i: [] for i in score_bins}
    all_scores_dict = {i: [] for i in score_bins}
    
    for _, (data, targets) in enumerate(val_loader):
        data, targets = data.to(device), targets.to(device)
        
        with torch.no_grad():
            
            logits = model(data)
            softmax = nn.functional.softmax(logits, dim=1)
            
            top_scores, preds = softmax.max(1)
            
            rare_idxs = (preds < num_classes/2).nonzero()
            common_idxs = (preds >= num_classes/2).nonzero()
            rare_top_scores = top_scores[rare_idxs].cpu().detach().numpy()
            common_top_scores = top_scores[common_idxs].cpu().detach().numpy()
            if is_isoreg:
                rare_top_scores = (np.expand_dims(rare_isoreg.predict(rare_top_scores), axis=1) if 
                        len(rare_idxs)>0 else np.empty(0)) 

                common_top_scores = (np.expand_dims(common_isoreg.predict(common_top_scores), axis=1) if 
                        len(common_idxs)>0 else np.empty(0))

            top_scores = np.append(rare_top_scores, common_top_scores)
            pred_bins = np.floor(10*top_scores)/10
            
            preds = preds[torch.cat((rare_idxs, common_idxs)).squeeze()]
            targets = targets[torch.cat((rare_idxs, common_idxs)).squeeze()]
            
            for i in range(len(targets)):
                key = round(pred_bins[i].item(), 2)
                all_scores_dict[key] += [targets[i]==preds[i]]

                if targets[i] < num_classes/2:
                    rare_scores_dict[key] += [targets[i]==preds[i]]
                
                else:
                    common_scores_dict[key] += [targets[i]==preds[i]]

    cal_rare_acc = torch.tensor(list(map(lambda x: sum(x)/len(x) if len(x)>0 else np.nan, list(rare_scores_dict.values()))))
    cal_common_acc = torch.tensor(list(map(lambda x: sum(x)/len(x) if len(x)>0 else np.nan, list(common_scores_dict.values()))))
    cal_all_acc = torch.tensor(list(map(lambda x: sum(x)/len(x) if len(x)>0 else np.nan, list(all_scores_dict.values()))))
    
    return cal_rare_acc, cal_common_acc, cal_all_acc

def train_isoreg(model, val_loader):
    print('isotonic regression')
    model.eval()

    top_scores = []
    preds_correct = []
    
    for batch_idx, (data, targets) in enumerate(val_loader):
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            logits = model(data)
            softmax = nn.functional.softmax(logits, dim=1)
            
            top_scores += [softmax.max(1).values,]
            
            preds = softmax.max(1).indices
            preds_correct += [(targets==preds).float(),]
    
    top_scores = torch.cat(top_scores, dim=0)
    preds_correct = torch.cat(preds_correct, dim=0)
    
    IsoReg = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip').fit(top_scores.cpu(), preds_correct.cpu())
    return IsoReg 

def get_embed_loader(data_dir,
                     class_dict,
                     num_classes,
                     embed_batch_size,
                     num_workers,
                     folder_to_class_file):

    data_transform = Compose([Resize((224, 224)),
                              ToTensor()])
                              
    dataset = ImageFolder(root=data_dir, transform=data_transform)
    
    with open(folder_to_class_file) as f:
        lines = list(zip(*[tuple(line.rstrip().split(' '))[:-1] for line in f]))
        keys = lines[0]
        vals = list(map(lambda x: int(x), lines[1]))
    
    temp_dict = dict(zip(keys, vals))
    
    idx_conv_dict = {}
    
    for folder_name in dataset.class_to_idx:
        if dataset.class_to_idx[folder_name] in list(class_dict.keys()):
            idx_conv_dict[dataset.class_to_idx[folder_name]] = temp_dict[folder_name]
    
    rare_classes = list(class_dict.keys())[:int(num_classes/2)]
    common_classes = list(class_dict.keys())[int(num_classes/2):]

    rare_idxs = torch.cat([(torch.tensor(dataset.targets)==cl).nonzero() for cl in rare_classes]).squeeze()
    common_idxs = torch.cat([(torch.tensor(dataset.targets)==cl).nonzero() for cl in common_classes]).squeeze()
    
    embed_idxs = torch.cat((rare_idxs, common_idxs))
    num_pts = len(embed_idxs)
    
    embed_data = Subset(dataset, embed_idxs)
    
    embed_loader = DataLoader(embed_data,
                              batch_size=embed_batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    return embed_loader, num_pts, idx_conv_dict

def get_datasets(embeds,
                 labels,
                 imbal,
                 num_init_pts,
                 num_classes):
    
    rare_idxs = (labels < 5).nonzero().squeeze()
    common_idxs = (labels >= 5).nonzero().squeeze()

    common_amount = len(common_idxs)
    rare_amount = int(np.floor(common_amount/imbal))
    
    rand_rare_idxs = rare_idxs[torch.randperm(len(rare_idxs))][:rare_amount]
    rand_common_idxs = common_idxs[torch.randperm(len(common_idxs))][:common_amount]
    imbal_idxs = torch.cat((rand_rare_idxs, rand_common_idxs))[torch.randperm(common_amount + rare_amount)]

    embeds, labels = embeds[imbal_idxs], labels[imbal_idxs]
    
    init_dataset, stream_dataset = random_split(TensorDataset(embeds, labels), [num_init_pts, len(imbal_idxs) - num_init_pts])

    return init_dataset, stream_dataset

def get_embeds(data_dir,
               class_dict,
               embed_dim,
               embed_batch_size,
               num_classes,
               num_workers,
               weights_path,
               embeds_path,
               embeds_labels_path,
               idx_conv_dict_path,
               folder_to_class_file,
               device):
    
    if not (file_exists(embeds_path) and
            file_exists(embeds_labels_path) and
            file_exists(idx_conv_dict_path)):
        
        embed_loader, num_pts, idx_conv_dict = get_embed_loader(data_dir,
                                                                class_dict,
                                                                num_classes,
                                                                embed_batch_size,
                                                                num_workers,
                                                                folder_to_class_file)

        pretrained_model = get_base_model(weights_path,
                                          num_classes,
                                          device)
        
        embed_model = Embed(pretrained_model)
        
        embed_model.eval()
        embeds = torch.zeros([num_pts, embed_dim])
        embeds_labels = torch.zeros([num_pts])
        
        print('embed_batch_size', embed_batch_size)
        
        with torch.no_grad():
            for idx, (data, targets) in enumerate(embed_loader):
                data = data.to(device)
                targets = torch.tensor([class_dict[y.item()] for y in targets]) 
                embeds[idx*embed_batch_size: min(num_pts, (idx+1)*embed_batch_size)] = embed_model(data).squeeze()
                embeds_labels[idx*embed_batch_size: min(num_pts, (idx+1)*embed_batch_size)] = targets
        
        torch.save(embeds, embeds_path)
        torch.save(embeds_labels, embeds_labels_path)
        torch.save(idx_conv_dict, idx_conv_dict_path)
    
    embeds = torch.load(embeds_path)
    embeds_labels = torch.load(embeds_labels_path)
    idx_conv_dict = torch.load(idx_conv_dict_path)
    
    return embeds, embeds_labels, idx_conv_dict

def get_test_embed_loader(embed_dim,
                          embed_batch_size,
                          batch_size,
                          num_classes,
                          num_workers,
                          weights_path,
                          test_dir,
                          test_embeds_path,
                          test_embeds_labels_path,
                          test_label_file,
                          class_dict,
                          num_test_pts,
                          idx_conv_dict,
                          device):
    
    if not (file_exists(test_embeds_path) and
            file_exists(test_embeds_labels_path)):
        
        test_loader = get_test_loader(test_dir,
                                      test_label_file,
                                      idx_conv_dict,
                                      batch_size,
                                      num_workers,
                                      class_dict)
        
        pretrained_model = get_base_model(weights_path,
                                          num_classes,
                                          device)
        
        embed_model = Embed(pretrained_model)
        
        embed_model.eval()
        
        test_embeds = torch.zeros([num_test_pts, embed_dim])
        test_embeds_labels = torch.zeros([num_test_pts])
        
        print('embed_batch_size', embed_batch_size)
        inv_dict = {idx_conv_dict[k]:k for k in idx_conv_dict}
        
        with torch.no_grad():
            for idx, (data, targets) in enumerate(test_loader):
                data = data.to(device)

                targets = torch.tensor([class_dict[inv_dict[y.item()]] for y in targets]) 
                test_embeds[idx*embed_batch_size: min(num_test_pts, (idx+1)*embed_batch_size)] = embed_model(data).squeeze()
                test_embeds_labels[idx*embed_batch_size: min(num_test_pts, (idx+1)*embed_batch_size)] = targets
        
        torch.save(test_embeds, test_embeds_path)
        torch.save(test_embeds_labels, test_embeds_labels_path)
    
    test_embeds = torch.load(test_embeds_path)
    test_embeds_labels = torch.load(test_embeds_labels_path)
    
    rare_embeds_idxs = (test_embeds_labels < num_classes/2).nonzero().squeeze(1)
    common_embeds_idxs = (test_embeds_labels >= num_classes/2).nonzero().squeeze(1)
    
    test_rare_embeds_idxs, val_rare_embeds_idxs = (rare_embeds_idxs[:int(len(rare_embeds_idxs)/2)],
                                                   rare_embeds_idxs[int(len(rare_embeds_idxs)/2):])

    test_common_embeds_idxs, val_common_embeds_idxs = (common_embeds_idxs[:int(len(common_embeds_idxs)/2)],
                                                       common_embeds_idxs[int(len(common_embeds_idxs)/2):])
    
    test_embeds_dataset = TensorDataset(test_embeds, test_embeds_labels)

    test_embeds_loader = DataLoader(Subset(test_embeds_dataset, torch.cat((test_rare_embeds_idxs, test_common_embeds_idxs))),
                                    batch_size=embed_batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)

    rare_val_embeds_loader = DataLoader(Subset(test_embeds_dataset, val_rare_embeds_idxs),
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True)
    
    common_val_embeds_loader = DataLoader(Subset(test_embeds_dataset, val_common_embeds_idxs),
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=True)
    
    val_embeds_loader = DataLoader(Subset(test_embeds_dataset, torch.cat((val_rare_embeds_idxs, val_common_embeds_idxs))),
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=True)
    
    return test_embeds_loader, rare_val_embeds_loader, common_val_embeds_loader, val_embeds_loader

def get_test_loader(test_dir,
                    test_label_file,
                    idx_conv_dict,
                    batch_size,
                    num_workers,
                    class_dict):

    data_transform = Compose([Resize((224, 224)),
                              ToTensor()])
    
    labels = [int(y) for y in genfromtxt(test_label_file)]
    
    test_dataset = ImageFolder(test_dir, transform=data_transform)
    test_dataset.samples = list(map(lambda x, y: (x[0], y), test_dataset.samples, labels))
    test_dataset.targets = labels
    
    class_idxs = torch.cat([(torch.tensor(test_dataset.targets)==cl).nonzero() for cl in list(idx_conv_dict.values())]).squeeze()
    test_sampler = sampler.SubsetRandomSampler(class_idxs)
    
    test_loader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=num_workers)

    return test_loader

def get_base_model(weights_path,
                   num_classes,
                   device):
    
    model = resnet50(pretrained=False).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    return model

class Embed(nn.Module):
    def __init__(self, model):
        super(Embed, self).__init__()
        self.embed = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        x = self.embed(x)
        return x

class LogRegModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogRegModel, self).__init__()
        self.linear= nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

def experiment(num_init_pts,
               imbals,
               unif_taus,
               dyn_taus,
               trials,
               num_sel_rounds,
               num_algs,
               num_agents,
               stream_size,
               num_test_pts,
               num_epochs,
               batch_size,
               num_workers,
               num_classes,
               dataset_name,
               num_sm_bins,
               is_isoreg,
               embed_batch_size,
               embed_dim,
               data_dir,
               test_dir,
               folder_to_class_file,
               test_label_file,
               rare_acc_path,
               all_acc_path,
               sizes_path,
               sum_sizes_path,
               weights_path,
               embeds_path,
               embeds_labels_path,
               idx_conv_dict_path,
               test_embeds_path,
               test_embeds_labels_path,
               model_path,
               device,
               budget,
               epsilon,
               seive_taus_path):
    
    if not file_exists(rare_acc_path):
        
        rare_acc=torch.zeros(len(trials),num_sel_rounds+1,num_algs)
        all_acc=torch.zeros(len(trials),num_sel_rounds+1,num_algs)
        
        sizes=torch.zeros(len(trials),num_sel_rounds+1,num_algs,num_classes)
        sum_sizes=torch.zeros(len(trials),num_sel_rounds+1,1)
        
        seive_taus=torch.zeros(len(trials),num_sel_rounds,2*num_agents)

        classes = random.sample(list(np.arange(1000)), num_classes)
        
        rare_classes = classes[:int(num_classes/2)]
        common_classes = classes[int(num_classes/2):]
        class_dict = dict(zip(rare_classes+common_classes, range(num_classes)))
        print('class_dict', class_dict)

        embeds, labels, idx_conv_dict = get_embeds(data_dir,
                                                   class_dict,
                                                   embed_dim,
                                                   embed_batch_size,
                                                   num_classes,
                                                   num_workers,
                                                   weights_path,
                                                   embeds_path,
                                                   embeds_labels_path,
                                                   idx_conv_dict_path,
                                                   folder_to_class_file,
                                                   device)
        
        test_embeds_loader, rare_val_embeds_loader, common_val_embeds_loader, val_embeds_loader = get_test_embed_loader(embed_dim,
                                                                                                                        embed_batch_size,
                                                                                                                        batch_size,
                                                                                                                        num_classes,
                                                                                                                        num_workers,
                                                                                                                        weights_path,
                                                                                                                        test_dir,
                                                                                                                        test_embeds_path,
                                                                                                                        test_embeds_labels_path,
                                                                                                                        test_label_file,
                                                                                                                        class_dict,
                                                                                                                        num_test_pts,
                                                                                                                        idx_conv_dict,
                                                                                                                        device)

        init_x = torch.empty(0)
        init_y = torch.empty(0)

        stream_datasets_dict = {key: None for key in range(num_agents)}

        for agent in range(num_agents):
            
            agent_init_dataset, agent_stream_dataset = get_datasets(embeds,
                                                                    labels,
                                                                    imbals[agent],
                                                                    num_init_pts,
                                                                    num_classes)

            agent_init_loader = DataLoader(agent_init_dataset,
                                           batch_size=num_init_pts,
                                           num_workers=num_workers,
                                           shuffle=True)

            agent_init_samples = enumerate(agent_init_loader)
            _, (agent_init_x, agent_init_y) = next(agent_init_samples)

            init_x = torch.cat((init_x, agent_init_x[:int(np.ceil(num_init_pts/num_agents))]))
            init_y = torch.cat((init_y, agent_init_y[:int(np.ceil(num_init_pts/num_agents))]))

            stream_datasets_dict[agent] = agent_stream_dataset

        print('init_props', [(init_y==i).sum() for i in range(num_classes)])

        sizes[:,0] = (torch.stack((torch.tensor([(init_y==i).sum() for i in range(num_classes)]),
                                   torch.tensor([(init_y==i).sum() for i in range(num_classes)]),
                                   torch.tensor([(init_y==i).sum() for i in range(num_classes)]),
                                   torch.tensor([(init_y==i).sum() for i in range(num_classes)]))))

        sum_sizes[:,0] = len(init_x)

        init_loader = DataLoader(TensorDataset(init_x, init_y),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True)

        model = LogRegModel(embed_dim, num_classes) 
        
        if file_exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)

        else:
            model = train(device,
                          num_epochs,
                          init_loader,
                          class_dict,
                          model)
            
            torch.save(model.state_dict(), model_path)

        rare_DIST_DMGT_UNIF_isoreg = train_isoreg(model, rare_val_embeds_loader) if is_isoreg else None
        common_DIST_DMGT_UNIF_isoreg = train_isoreg(model, common_val_embeds_loader) if is_isoreg else None
        rare_DIST_DMGT_DYN_isoreg = train_isoreg(model, rare_val_embeds_loader) if is_isoreg else None
        common_DIST_DMGT_DYN_isoreg = train_isoreg(model, common_val_embeds_loader) if is_isoreg else None
        rare_SEIVE_isoreg = train_isoreg(model, rare_val_embeds_loader) if is_isoreg else None
        common_SEIVE_isoreg = train_isoreg(model, common_val_embeds_loader) if is_isoreg else None
        
        rare_acc[:,0] = (

                torch.cat((calc_acc(model, test_embeds_loader, num_classes)[0], 
                           calc_acc(model, test_embeds_loader, num_classes)[0],
                           calc_acc(model, test_embeds_loader, num_classes)[0],
                           calc_acc(model, test_embeds_loader, num_classes)[0])))
        
        all_acc[:,0] = (
                
                torch.cat((calc_acc(model, test_embeds_loader, num_classes)[1],
                           calc_acc(model, test_embeds_loader, num_classes)[1],
                           calc_acc(model, test_embeds_loader, num_classes)[1],
                           calc_acc(model, test_embeds_loader, num_classes)[1])))
        
        for trial in trials:
            print('trial', trial)
            
            DIST_DMGT_UNIF_model = load_model(model, embed_dim, num_classes, device)
            DIST_DMGT_DYN_model = load_model(model, embed_dim, num_classes, device)
            RAND_model = load_model(model, embed_dim, num_classes, device)
            SEIVE_model = load_model(model, embed_dim, num_classes, device)

            stream_loaders_dict = {agent: DataLoader(stream_datasets_dict[agent],
                                                     batch_size=stream_size,
                                                     num_workers=num_workers,
                                                     shuffle=True) for agent in range(num_agents)}
            
            stream_samples_dict = {agent: enumerate(stream_loaders_dict[agent]) for agent in range(num_agents)}
            
            for sel_round in range(num_sel_rounds):
                print('sel_round', sel_round)

                DIST_DMGT_UNIF_x = torch.empty(0)
                DIST_DMGT_UNIF_y = torch.empty(0)
                DIST_DMGT_DYN_x = torch.empty(0)
                DIST_DMGT_DYN_y = torch.empty(0)
                RAND_x = torch.empty(0)
                RAND_y = torch.empty(0)
                SEIVE_x = torch.empty(0)
                SEIVE_y = torch.empty(0)
                seive_min_max_taus = torch.empty(0)

                for agent in range(num_agents):
                    print('agent', agent)

                    _, (agent_stream_x, agent_stream_y) = next(stream_samples_dict[agent])

                    agent_DIST_DMGT_UNIF_x, agent_DIST_DMGT_UNIF_y, agent_RAND_x, agent_RAND_y = get_DIST_DMGT_subsets(agent_stream_x,
                                                                                                             agent_stream_y,
                                                                                                             unif_taus,
                                                                                                             sel_round,
                                                                                                             DIST_DMGT_UNIF_model,
                                                                                                             num_classes,
                                                                                                             is_isoreg,
                                                                                                             rare_DIST_DMGT_UNIF_isoreg,
                                                                                                             common_DIST_DMGT_UNIF_isoreg,
                                                                                                             device,
                                                                                                             budget)
                    
                    agent_DIST_DMGT_DYN_x, agent_DIST_DMGT_DYN_y, _, _ = get_DIST_DMGT_subsets(agent_stream_x,
                                                                                                             agent_stream_y,
                                                                                                             dyn_taus,
                                                                                                             sel_round,
                                                                                                             DIST_DMGT_DYN_model,
                                                                                                             num_classes,
                                                                                                             is_isoreg,
                                                                                                             rare_DIST_DMGT_DYN_isoreg,
                                                                                                             common_DIST_DMGT_DYN_isoreg,
                                                                                                             device,
                                                                                                             budget)
                    
                    agent_SEIVE_x, agent_SEIVE_y, agent_seive_min_max_taus = get_SEIVE_subsets(agent_stream_x,
                                                                                               agent_stream_y,
                                                                                               SEIVE_model,
                                                                                               num_classes,
                                                                                               is_isoreg,
                                                                                               rare_SEIVE_isoreg,
                                                                                               common_SEIVE_isoreg,
                                                                                               device,
                                                                                               budget,
                                                                                               epsilon)
                    
                    DIST_DMGT_UNIF_x = torch.cat((DIST_DMGT_UNIF_x, agent_DIST_DMGT_UNIF_x))
                    DIST_DMGT_UNIF_y = torch.cat((DIST_DMGT_UNIF_y, agent_DIST_DMGT_UNIF_y))
                    DIST_DMGT_DYN_x = torch.cat((DIST_DMGT_DYN_x, agent_DIST_DMGT_DYN_x))
                    DIST_DMGT_DYN_y = torch.cat((DIST_DMGT_DYN_y, agent_DIST_DMGT_DYN_y))
                    RAND_x = torch.cat((RAND_x, agent_RAND_x))
                    RAND_y = torch.cat((RAND_y, agent_RAND_y))
                    SEIVE_x = torch.cat((SEIVE_x, agent_SEIVE_x))
                    SEIVE_y = torch.cat((SEIVE_y, agent_SEIVE_y))
                    seive_min_max_taus = torch.cat((seive_min_max_taus, agent_seive_min_max_taus))
                
                seive_taus[trial,sel_round] = seive_min_max_taus

                sizes[trial,sel_round+1] = (

                        sizes[trial,sel_round] + 
                        torch.stack((torch.tensor([(DIST_DMGT_UNIF_y==i).sum() for i in range(num_classes)]),
                                     torch.tensor([(DIST_DMGT_DYN_y==i).sum() for i in range(num_classes)]),
                                     torch.tensor([(RAND_y==i).sum() for i in range(num_classes)]),
                                     torch.tensor([(SEIVE_y==i).sum() for i in range(num_classes)]))))

                sum_sizes[trial,sel_round+1] = (

                        torch.tensor([sum_sizes[trial,sel_round] + len(DIST_DMGT_UNIF_y)]))
                    
                DIST_DMGT_UNIF_model = train(device,
                                        num_epochs,
                                        DataLoader(TensorDataset(DIST_DMGT_UNIF_x, DIST_DMGT_UNIF_y),
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True),
                                        class_dict,
                                        DIST_DMGT_UNIF_model)

                DIST_DMGT_DYN_model = train(device,
                                        num_epochs,
                                        DataLoader(TensorDataset(DIST_DMGT_DYN_x, DIST_DMGT_DYN_y),
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True),
                                        class_dict,
                                        DIST_DMGT_DYN_model)
                
                RAND_model = train(device,
                                   num_epochs,
                                   DataLoader(TensorDataset(RAND_x, RAND_y),
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=True),
                                   class_dict,
                                   RAND_model)
                
                SEIVE_model = train(device,
                                    num_epochs,
                                    DataLoader(TensorDataset(SEIVE_x, SEIVE_y),
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True),
                                    class_dict,
                                    SEIVE_model)
                
                rare_DIST_DMGT_UNIF_isoreg = train_isoreg(DIST_DMGT_UNIF_model, rare_val_embeds_loader) if is_isoreg else None
                common_DIST_DMGT_UNIF_isoreg = train_isoreg(DIST_DMGT_UNIF_model, common_val_embeds_loader) if is_isoreg else None
                rare_DIST_DMGT_DYN_isoreg = train_isoreg(DIST_DMGT_DYN_model, rare_val_embeds_loader) if is_isoreg else None
                common_DIST_DMGT_DYN_isoreg = train_isoreg(DIST_DMGT_DYN_model, common_val_embeds_loader) if is_isoreg else None
                rare_SEIVE_isoreg = train_isoreg(SEIVE_model, rare_val_embeds_loader) if is_isoreg else None
                rare_SEIVE_isoreg = train_isoreg(SEIVE_model, rare_val_embeds_loader) if is_isoreg else None
                
                rare_acc[trial,sel_round+1] = (

                        torch.cat((calc_acc(DIST_DMGT_UNIF_model, test_embeds_loader, num_classes)[0], 
                                   calc_acc(DIST_DMGT_DYN_model, test_embeds_loader, num_classes)[0],
                                   calc_acc(RAND_model, test_embeds_loader, num_classes)[0],
                                   calc_acc(SEIVE_model, test_embeds_loader, num_classes)[0])))
                
                all_acc[trial,sel_round+1] = (
                        
                        torch.cat((calc_acc(DIST_DMGT_UNIF_model, test_embeds_loader, num_classes)[1],
                                   calc_acc(DIST_DMGT_DYN_model, test_embeds_loader, num_classes)[1],
                                   calc_acc(RAND_model, test_embeds_loader, num_classes)[1],
                                   calc_acc(SEIVE_model, test_embeds_loader, num_classes)[1])))
            
        torch.save(rare_acc, rare_acc_path)
        torch.save(all_acc, all_acc_path)

        torch.save(sizes, sizes_path)
        torch.save(sum_sizes, sum_sizes_path)

        torch.save(seive_taus, seive_taus_path)
        
    rare_acc = torch.load(rare_acc_path)
    all_acc = torch.load(all_acc_path)

    sizes = torch.load(sizes_path)
    sum_sizes = torch.load(sum_sizes_path)

    seive_taus = torch.load(seive_taus_path)
    
    return rare_acc, all_acc, sizes, sum_sizes, seive_taus

def dataframe(data,
              trials,
              num_sel_rounds):

    rare_acc, all_acc, sizes, sum_sizes, seive_taus = data
    df = pd.DataFrame(columns=['trial',
                               'sel_rnd',
                               'DIST_DMGT_UNIF_all_acc',
                               'DIST_DMGT_DYN_all_acc',
                               'RAND_all_acc',
                               'SEIVE_all_acc',
                               'DIST_DMGT_UNIF_rare_acc',
                               'DIST_DMGT_DYN_rare_acc',
                               'RAND_rare_acc',
                               'SEIVE_rare_acc',
                               'DIST_DMGT_UNIF_rare_amnt',
                               'DIST_DMGT_DYN_rare_amnt',
                               'RAND_rare_amnt',
                               'SEIVE_rare_amnt',
                               'DIST_DMGT_UNIF_common_amnt',
                               'DIST_DMGT_DYN_common_amnt',
                               'RAND_common_amnt',
                               'SEIVE_common_amnt',
                               'sum_sizes',
                               'sum_sizes_perc'])
    
    for trial in trials:
        df = df.append(pd.DataFrame({'trial':trial*torch.ones(num_sel_rounds+1),
                                     'sel_rnd':torch.arange(num_sel_rounds+1),
                                     'DIST_DMGT_UNIF_all_acc':all_acc[trial,:,0].squeeze(),
                                     'DIST_DMGT_DYN_all_acc':all_acc[trial,:,1].squeeze(),
                                     'RAND_all_acc':all_acc[trial,:,2].squeeze(),
                                     'SEIVE_all_acc':all_acc[trial,:,3].squeeze(),
                                     'DIST_DMGT_UNIF_rare_acc':rare_acc[trial,:,0].squeeze(),
                                     'DIST_DMGT_DYN_rare_acc':rare_acc[trial,:,1].squeeze(),
                                     'RAND_rare_acc':rare_acc[trial,:,2].squeeze(),
                                     'SEIVE_rare_acc':rare_acc[trial,:,3].squeeze(),
                                     'DIST_DMGT_UNIF_rare_amnt':(torch.stack([x[:5].sum().int() for x in sizes[trial,:,0]])),
                                     'DIST_DMGT_DYN_rare_amnt':(torch.stack([x[:5].sum().int() for x in sizes[trial,:,1]])),
                                     'RAND_rare_amnt':(torch.stack([x[:5].sum().int() for x in sizes[trial,:,2]])),
                                     'SEIVE_rare_amnt':(torch.stack([x[:5].sum().int() for x in sizes[trial,:,3]])),
                                     'DIST_DMGT_UNIF_common_amnt':(torch.stack([x[5:].sum().int() for x in sizes[trial,:,0]])),
                                     'DIST_DMGT_DYN_common_amnt':(torch.stack([x[5:].sum().int() for x in sizes[trial,:,1]])),
                                     'RAND_common_amnt':(torch.stack([x[5:].sum().int() for x in sizes[trial,:,2]])),
                                     'SEIVE_common_amnt':(torch.stack([x[5:].sum().int() for x in sizes[trial,:,3]])),
                                     'sum_sizes':sum_sizes[trial].squeeze().int(),
                                     'sum_sizes_perc':(sum_sizes[trial].squeeze().int()/(30*(num_sel_rounds+1)))}),
                                     ignore_index=True)
    
    return df

def balance_plot(date_time,sizes,num_algs,num_classes,num_sel_rounds,unif_taus,dyn_taus):
    alg_names = ['Dist. DMGT w/ Uniform Thresholds','Dist. DMGT w/ Increasing Thresholds','RAND','SIEVE']
    avg_sizes = sizes.mean(dim=0)
    non_cum_sizes = torch.zeros(num_algs, num_classes, num_sel_rounds)
    for i in range(num_algs):
        for j in range(num_classes):
            per_class_sizes = list(avg_sizes.transpose(0,1).transpose(1,2)[i,j])
            non_cum_sizes[i,j] = torch.tensor([per_class_sizes[i+1]-per_class_sizes[i] for i in range(num_sel_rounds)])
    
    non_cum_sizes = non_cum_sizes.transpose(1,2).transpose(0,1)

    rare_avg_sizes = non_cum_sizes[:,:,:5].mean(dim=2)
    common_avg_sizes = non_cum_sizes[:,:,5:].mean(dim=2)

    rare_common_avg_sizes = torch.cat((rare_avg_sizes.unsqueeze(2), common_avg_sizes.unsqueeze(2)),dim=2)
    fig, ax = plt.subplots()
    for i in range(num_algs):
        sns.lineplot(x=np.arange(num_sel_rounds),
                     y=rare_common_avg_sizes[:,i,0],
                     color=sns.color_palette('pastel')[i],
                     label=alg_names[i] + ' rare classes')
        sns.lineplot(x=np.arange(num_sel_rounds),
                     y=rare_common_avg_sizes[:,i,1],
                     color=sns.color_palette('muted')[i],
                     label=alg_names[i] + ' common classes',
                     linestyle='--')
    
    n = Symbol('n')
    unif_balanced_size = int(np.ceil(solve(sympy.sqrt(n+1) - sympy.sqrt(n) - unif_taus[0], n)[0]))
    sns.lineplot(x=np.arange(num_sel_rounds),
                 y=unif_balanced_size,
                 color='gray',
                 label=r'Balanced Uniform $\tau$', 
                 linestyle='--')
    
    n = Symbol('n')
    dyn_balanced_sizes = [int(np.ceil(solve(sympy.sqrt(n+1) - sympy.sqrt(n) - dyn_taus[i], n)[0])) for i in range(len(dyn_taus))]
    sns.lineplot(x=np.arange(num_sel_rounds),
                 y=dyn_balanced_sizes,
                 color='black',
                 label=r'Balanced Increasing $\tau$',
                 linestyle='--')
    
    ax.set_xlabel('Selection Round')
    ax.set_ylabel('Average Number of Points per Class')
    
    ax.legend(fontsize=5,ncol=2)
    fig.tight_layout()
    sns.despine()
    fig.savefig(img_dir + date_time + '_balance.pdf')

def accuracy_plot(df, date_time, num_algs, num_sel_rounds):
    
    fig, ax = plt.subplots()
    sns.despine()
    
    all_acc_data_files = ['DIST_DMGT_UNIF_all_acc','DIST_DMGT_DYN_all_acc','RAND_all_acc','SEIVE_all_acc'] 
    rare_acc_data_files = ['DIST_DMGT_UNIF_rare_acc','DIST_DMGT_DYN_rare_acc','RAND_rare_acc','SEIVE_rare_acc']
    
    all_acc_labels = ['Dist. DMGT w/ Uniform Thresholds: all classes',
                      'Dist. DMGT w/ Increasing Thresholds: all classes',
                      'RAND: all classes',
                      'SIEVE: all classes']
    
    rare_acc_labels = ['Dist. DMGT w/ Uniform Thresholds: rare classes',
                       'Dist. DMGT w/ Increasing Thresholds: rare classes',
                       'RAND: rare classes',
                       'SIEVE: rare classes']

    for i in range(num_algs):
        sns.lineplot(data=df[['sel_rnd', all_acc_data_files[i]]],
                     x='sel_rnd',
                     y=all_acc_data_files[i],
                     color=sns.color_palette('muted')[i],
                     ci=95,
                     estimator='mean',
                     label=all_acc_labels[i],
                     marker='*',
                     markerfacecolor='black',
                     linestyle='--',
                     markersize=8)

        sns.lineplot(data=df[['sel_rnd', rare_acc_data_files[i]]],
                     x='sel_rnd',
                     y=rare_acc_data_files[i],
                     color=sns.color_palette('pastel')[i],
                     ci=95,
                     estimator='mean',
                     label=rare_acc_labels[i],
                     marker='*',
                     markerfacecolor='black',
                     markersize=8)

    sizes_labels = [1000]
    amount_cols = ['DIST_DMGT_UNIF_common_amnt', 'DIST_DMGT_DYN_rare_amnt',
                   'DIST_DMGT_DYN_common_amnt', 'DIST_DMGT_DYN_rare_amnt',
                   'RAND_common_amnt','RAND_rare_amnt',
                   'SEIVE_common_amnt','SEIVE_rare_amnt']

    #for i in np.arange(1,num_sel_rounds+1):
    #    sizes_labels += [int(np.floor(df[df['sel_rnd']==i][amount_cols].mean()))]
    #
    #ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,num_sel_rounds+1)))
    #ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{i}'+'\n'+f'{sizes_labels[i]}' for i in range(num_sel_rounds)]))
    
    ax.legend(fontsize=6)
    ax.set_xlabel('Selection Round')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.savefig(img_dir + date_time + '_accuracy.pdf')
    
#def extract_data(orig_dir, data_dir):
#    for root, dirs, _ in os.walk(orig_dir):
#        for name in dirs:
#            if name.startswith('n'):
#                os.symlink(os.path.join(root, name), os.path.join(data_dir, name))
#

parser = argparse.ArgumentParser()
parser.add_argument('--num_init_pts', type=int, default=1000)
parser.add_argument('--imbals', nargs='+', type=int, default=[2,5,10])
parser.add_argument('--unif_taus', nargs='+', type=float, default=6*[0.1])
#parser.add_argument('--dyn_taus', nargs='+', type=float, default=[0.05,0.05,0.1,0.1,0.15,0.15])
parser.add_argument('--dyn_taus', nargs='+', type=float, default=[0.1,0.1,0.13,0.13,0.15,0.15])
parser.add_argument('--trials', nargs='+', type=int, default=np.arange(5))
parser.add_argument('--num_sel_rounds', type=int, default=6)
parser.add_argument('--num_algs', type=int, default=4)
parser.add_argument('--num_agents', type=int, default=3)
parser.add_argument('--stream_size', type=int, default=500)
parser.add_argument('--num_test_pts', type=int, default=500)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--dataset_name', type=str, default='IMNET')
parser.add_argument('--num_sm_bins', type=int, default=11) 
parser.add_argument('--is_isoreg', type=bool, default=True)
parser.add_argument('--embed_batch_size', type=int, default=256)
parser.add_argument('--embed_dim', type=int, default=2048)
parser.add_argument('--data_dir', type=str, default='/home/eecs/mwerner/imagenet/train/')
parser.add_argument('--test_dir', type=str, default='/work/data/imagenet/val/')
parser.add_argument('--folder_to_class_file', type=str, default='/home/eecs/mwerner/IMNET_CODE/map_clsloc.txt')
parser.add_argument('--test_label_file', type=str, default='/home/eecs/mwerner/IMNET_CODE/test_labels.txt')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--budget', type=int, default=250)
parser.add_argument('--epsilon', type=float, default=0.1)

if __name__ == "__main__":
    
    args = parser.parse_args()
    device=torch.device('cuda:0') 
    print('ISOREG', args.is_isoreg) 

    # fix randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed) 
    torch.backends.cudnn.deterministic=True
   
    img_dir = f'/home/eecs/mwerner/DMGT_neurips_response/{args.dataset_name}_CODE/output_files/imgs/{date_time}/'
    #os.mkdir(img_dir)
    
    val_dir = f'/home/eecs/mwerner/DMGT_neurips_response/{args.dataset_name}_CODE/output_files/vals/{date_time}/'
    #os.mkdir(val_dir)
    
    # embeds paths
    embeds_dir = f'/home/eecs/mwerner/DMGT_neurips_response/{args.dataset_name}_CODE/embeds/'

    embeds_path=embeds_dir + f'fed_dmgt_embeds_1x_{args.embed_dim}.pkl'
    embeds_labels_path=embeds_dir + 'fed_dmgt_labels_1x.pkl'
    idx_conv_dict_path=embeds_dir + 'fed_dmgt_idx_conv_dict.pkl'
    test_embeds_path=embeds_dir + f'fed_dmgt_test_embeds_1x_{args.embed_dim}.pkl'
    test_embeds_labels_path=embeds_dir + f'fed_dmgt_test_labels_1x.pkl'
    
    # simclr weights path
    weights_dir = f'/home/eecs/mwerner/{args.dataset_name}_CODE/weights_dir/'
    
    weights_path=weights_dir + 'resnet50-1x.pth'

    # model path
    model_path=f'/home/eecs/mwerner/DMGT_neurips_response/{args.dataset_name}_CODE/models/fed_model.pkl'

    # accuracy paths
    all_acc_path=val_dir + 'all_acc.pkl'
    rare_acc_path=val_dir + 'rare_acc.pkl'

    # labeled set sizes paths
    sizes_path=val_dir + 'sizes.pkl'
    sum_sizes_path=val_dir + 'sum_sizes.pkl'
    
    # min and max thresholds from seive,seive++ algorithm
    seive_taus_path=val_dir + 'seive_taus.pkl'
    seive_plus_taus_path=val_dir + 'seive_plus_taus.pkl'

    input_args = [args.num_init_pts,
                  args.imbals,
                  args.unif_taus,
                  args.dyn_taus,
                  args.trials,
                  args.num_sel_rounds,
                  args.num_algs,
                  args.num_agents,
                  args.stream_size,
                  args.num_test_pts,
                  args.num_epochs,
                  args.batch_size,
                  args.num_workers,
                  args.num_classes,
                  args.dataset_name,
                  args.num_sm_bins,
                  args.is_isoreg,
                  args.embed_batch_size,
                  args.embed_dim,
                  args.data_dir,
                  args.test_dir,
                  args.folder_to_class_file,
                  args.test_label_file,
                  rare_acc_path,
                  all_acc_path,
                  sizes_path,
                  sum_sizes_path,
                  weights_path,
                  embeds_path,
                  embeds_labels_path,
                  idx_conv_dict_path,
                  test_embeds_path,
                  test_embeds_labels_path,
                  model_path,
                  device,
                  args.budget,
                  args.epsilon,
                  seive_taus_path]
    
    rare_acc, all_acc, sizes, sum_sizes, seive_taus = experiment(*input_args)
    
    df = dataframe(experiment(*input_args),
                   args.trials,
                   args.num_sel_rounds)
    
    #balance_plot(date_time,sizes,args.num_algs,args.num_classes,args.num_sel_rounds,args.unif_taus,args.dyn_taus)
    accuracy_plot(df,date_time,args.num_algs,args.num_sel_rounds)
