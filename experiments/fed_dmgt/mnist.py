import numpy as np
import random
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import os
from os.path import exists as file_exists
import argparse
from ../helper_funcs import class_card, get_subsets
from ../model_funcs import MnistResNet, train, load_model, calc_acc, train_isoreg
from ../data_funcs/mnist import get_datasets, get_val_loaders

def experiment(num_init_pts,
               imbals,
               tau,
               trials,
               num_agents,
               num_algs,
               stream_size,
               num_test_pts,
               num_epochs,
               batch_size,
               num_workers,
               num_classes,
               device,
               num_sel_rnds):
        
    rare_acc=torch.zeros(len(trials),num_sel_rnds+1,num_algs)
    all_acc=torch.zeros(len(trials),num_sel_rnds+1,num_algs)
    
    sizes=torch.zeros(len(trials),num_sel_rnds+1,num_algs,num_classes)
    sum_sizes=torch.zeros(len(trials),num_sel_rnds+1,1)
    
    test_loader, rare_val_loader, common_val_loader, val_loader = get_val_loaders(num_test_pts, batch_size, num_workers, num_classes)
    
    init_x = torch.empty(0)
    init_y = torch.empty(0)
    
    stream_datasets_dict = {key: None for key in range(num_agents)}
    
    for agent in range(num_agents):
        agent_init_dataset, agent_stream_dataset = get_datasets(num_init_pts, imbals[agent], num_classes)
        agent_init_loader = DataLoader(agent_init_dataset, batch_size=num_init_pts, num_workers=num_workers, shuffle=True)

        agent_init_samples = enumerate(agent_init_loader)
        _, (agent_init_x, agent_init_y) = next(agent_init_samples)
        
        init_x = torch.cat((init_x, agent_init_x[:int(np.ceil(num_init_pts/num_agents))]))
        init_y = torch.cat((init_y, agent_init_y[:int(np.ceil(num_init_pts/num_agents))]))
        
        stream_datasets_dict[agent] = agent_stream_dataset
    
    sizes[:,0] = (torch.stack((torch.tensor([(init_y==i).float().sum() for i in range(num_classes)]),
                               torch.tensor([(init_y==i).float().sum() for i in range(num_classes)]))))
            
    sum_sizes[:,0] = len(init_x)
                
    init_loader = DataLoader(TensorDataset(init_x, init_y), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = MnistResNet()
    model = train(device, num_epochs, init_loader, model)

    rare_isoreg = train_isoreg(model, rare_val_loader)
    common_isoreg = train_isoreg(model, common_val_loader)
    
    rare_acc[:,0] = (

        torch.cat((calc_acc(model, test_loader, num_classes)[0], 
                   calc_acc(model, test_loader, num_classes)[0])))
    
    all_acc[:,0] = (
            
        torch.cat((calc_acc(model, test_loader, num_classes)[1],
                   calc_acc(model, test_loader, num_classes)[1])))
    
    for trial in trials: 
        FED_DMGT_model = load_model(model, device)
        RAND_model = load_model(model, device)
        
        stream_loaders_dict = {agent: DataLoader(stream_datasets_dict[agent],
                                                 batch_size=stream_size,
                                                 num_workers=num_workers,
                                                 shuffle=True) for agent in range(num_agents)}

        stream_samples_dict = {agent: enumerate(stream_loaders_dict[agent]) for agent in range(num_agents)}
        
        for sel_rnd in range(num_sel_rnds):
            FED_DMGT_x = torch.empty(0)
            FED_DMGT_y = torch.empty(0)
            RAND_x = torch.empty(0)
            RAND_y = torch.empty(0)
            
            for agent in range(num_agents):
                _, (agent_stream_x, agent_stream_y) = next(stream_samples_dict[agent])
                agent_FED_DMGT_x, agent_FED_DMGT_y, agent_RAND_x, agent_RAND_y = get_subsets(agent_stream_x,
                                                                                             agent_stream_y,
                                                                                             tau,
                                                                                             FED_DMGT_model,
                                                                                             num_classes,
                                                                                             rare_isoreg,
                                                                                             common_isoreg,
                                                                                             device)
                    
                
                FED_DMGT_x = torch.cat((FED_DMGT_x, agent_FED_DMGT_x))                
                FED_DMGT_y = torch.cat((FED_DMGT_y, agent_FED_DMGT_y))
                RAND_x = torch.cat((RAND_x, agent_RAND_x))                
                RAND_y = torch.cat((RAND_y, agent_RAND_y))
            
            sizes[trial,sel_rnd+1] = (
                   
                  sizes[trial,sel_rnd] + 
                  torch.stack((torch.tensor([(FED_DMGT_y==i).float().sum() for i in range(num_classes)]),
                               torch.tensor([(RAND_y==i).float().sum() for i in range(num_classes)]))))
                  
            sum_sizes[trial,sel_rnd+1] = (
                
                      torch.tensor([sum_sizes[trial,sel_rnd] + len(FED_DMGT_y)]))
            
            FED_DMGT_model = train(device,
                                   num_epochs,
                                   DataLoader(TensorDataset(FED_DMGT_x, FED_DMGT_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                                   FED_DMGT_model)
            
            RAND_model = train(device,
                               num_epochs,
                               DataLoader(TensorDataset(RAND_x, RAND_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                               RAND_model)
                        
            rare_isoreg = train_isoreg(FED_DMGT_model, rare_val_loader)
            common_isoreg = train_isoreg(FED_DMGT_model, common_val_loader)
    
            rare_acc[trial,sel_rnd+1] = (

                     torch.cat((calc_acc(FED_DMGT_model, test_loader, num_classes)[0], 
                                calc_acc(RAND_model, test_loader, num_classes)[0])))
            
            all_acc[trial,sel_rnd+1] = (
                    
                    torch.cat((calc_acc(FED_DMGT_model, test_loader, num_classes)[1],
                               calc_acc(RAND_model, test_loader, num_classes)[1])))

    return rare_acc, all_acc, sizes, sum_sizes

