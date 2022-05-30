# import libraries
import numpy as np
import random
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import os
from os.path import exists as file_exists
import argparse
# import internal functions
from ..exp_utils import class_card, get_subsets, MnistResNet, train, load_model, calc_acc, train_isoreg
from ..data_utils.mnist_data_utils import get_datasets, get_val_loaders
from ..dataframes import dist_dmgt_df

# main experiment -- runs DMGT and RAND; generates all data for figures
def experiment(num_init_pts,
               imbals,
               taus,
               trials,
               num_agents,
               num_algs,
               stream_size,
               num_epochs,
               batch_size,
               num_workers,
               num_classes,
               device,
               data_path,
               num_sel_rnds,
               num_test_pts):
        
    rare_acc=torch.zeros(len(trials),num_sel_rnds+1,num_algs)
    all_acc=torch.zeros(len(trials),num_sel_rnds+1,num_algs)
    
    sizes=torch.zeros(len(trials),num_sel_rnds+1,num_algs,num_classes)
    sum_sizes=torch.zeros(len(trials),num_sel_rnds+1,1)
    
    test_loader, rare_val_loader, common_val_loader, val_loader = get_val_loaders(num_test_pts, batch_size, num_workers, num_classes, data_path)
    
    init_x = torch.empty(0)
    init_y = torch.empty(0)
    
    stream_datasets_dict = {key: None for key in range(num_agents)}
    
    for agent in range(num_agents):
        agent_init_dataset, agent_stream_dataset = get_datasets(num_init_pts, imbals[agent], num_classes, data_path)
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

    rare_isoreg = train_isoreg(model, rare_val_loader, device)
    common_isoreg = train_isoreg(model, common_val_loader, device)
    
    rare_acc[:,0] = (
        torch.cat((calc_acc(model, test_loader, num_classes, device)[0], 
                   calc_acc(model, test_loader, num_classes, device)[0])))
    
    all_acc[:,0] = (
        torch.cat((calc_acc(model, test_loader, num_classes, device)[1],
                   calc_acc(model, test_loader, num_classes, device)[1])))
    
    for trial in trials:
        DIST_DMGT_model = load_model(model, device)
        RAND_model = load_model(model, device)
        
        stream_loaders_dict = {agent: DataLoader(stream_datasets_dict[agent],
                                                 batch_size=stream_size,
                                                 num_workers=num_workers,
                                                 shuffle=True) for agent in range(num_agents)}

        stream_samples_dict = {agent: enumerate(stream_loaders_dict[agent]) for agent in range(num_agents)}
        
        for sel_rnd in range(num_sel_rnds):
            DIST_DMGT_x = torch.empty(0)
            DIST_DMGT_y = torch.empty(0)
            RAND_x = torch.empty(0)
            RAND_y = torch.empty(0)
            
            ## replace previous two lines with these two if running METAFED-DMGT
            # stream_x = torch.empty(0)
            # stream_y = torch.empty(0)
            
            for agent in range(num_agents):
                tau = taus[agent]
                _, (agent_stream_x, agent_stream_y) = next(stream_samples_dict[agent])
                agent_DIST_DMGT_x, agent_DIST_DMGT_y, agent_RAND_x, agent_RAND_y = get_subsets(agent_stream_x,
                                                                                               agent_stream_y,
                                                                                               tau,
                                                                                               DIST_DMGT_model,
                                                                                               num_classes,
                                                                                               rare_isoreg,
                                                                                               common_isoreg,
                                                                                               device)
                    
                
                DIST_DMGT_x = torch.cat((DIST_DMGT_x, agent_DIST_DMGT_x))                
                DIST_DMGT_y = torch.cat((DIST_DMGT_y, agent_DIST_DMGT_y))
                RAND_x = torch.cat((RAND_x, agent_RAND_x))                
                RAND_y = torch.cat((RAND_y, agent_RAND_y))
            
                ## replace previous two lines with these two if running Distributed DMGT w/ Filtering
                
                # stream_x = torch.cat((stream_x, agent_stream_x))
                # stream_y = torch.cat((stream_y, agent_stream_y))
            
            ## uncomment the following lines if running Distributed DMGT w/ Filtering
            
            # DIST_DMGT_x, DIST_DMGT_y = get_subsets(DIST_DMGT_x, DIST_DMGT_y, tau, DIST_DMGT_model, num_classes, rare_isoreg, common_isoreg, device)
            # rand_idxs = torch.randperm(len(stream_x))[:len(DIST_DMGT_x)]
            # RAND_x = stream_x[rand_idxs]
            # RAND_y = stream_y[rand_idxs]
            
            sizes[trial,sel_rnd+1] = (
                   
                  sizes[trial,sel_rnd] + 
                  torch.stack((torch.tensor([(DIST_DMGT_y==i).float().sum() for i in range(num_classes)]),
                               torch.tensor([(RAND_y==i).float().sum() for i in range(num_classes)]))))
                  
            sum_sizes[trial,sel_rnd+1] = (
                
                      torch.tensor([sum_sizes[trial,sel_rnd] + len(DIST_DMGT_y)]))
            
            DIST_DMGT_model = train(device,
                                    num_epochs,
                                    DataLoader(TensorDataset(DIST_DMGT_x, DIST_DMGT_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                                    DIST_DMGT_model)
            
            RAND_model = train(device,
                               num_epochs,
                               DataLoader(TensorDataset(RAND_x, RAND_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                               RAND_model)
                        
            rare_isoreg = train_isoreg(DIST_DMGT_model, rare_val_loader, device)
            common_isoreg = train_isoreg(DIST_DMGT_model, common_val_loader, device)
    
            rare_acc[trial,sel_rnd+1] = (
                     torch.cat((calc_acc(DIST_DMGT_model, test_loader, num_classes, device)[0], 
                                calc_acc(RAND_model, test_loader, num_classes, device)[0])))
            
            all_acc[trial,sel_rnd+1] = (
                    torch.cat((calc_acc(DIST_DMGT_model, test_loader, num_classes, device)[1],
                               calc_acc(RAND_model, test_loader, num_classes, device)[1])))
    
    data = rare_acc, all_acc, sizes, sum_sizes
    df = dist_dmgt_df(data, trials, num_sel_rnds)
    
    return df

