# import libraries
import numpy as np
from numpy import genfromtxt
import random
import torch 
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50
import pandas as pd
from os.path import exists as file_exists
import argparse
# import internal functions
from ..exp_utils import class_card, get_subsets, Embed, LogRegModel, train, load_model, calc_acc, train_isoreg
from ..data_utils/imnet_data_utils import get_embed_loader, get_embeds, get_test_embed_loaders, get_test_loader
from ..dataframes import fed_dmgt_df

# main experiment -- runs DMGT and RAND; generates all data for figures
def experiment(num_init_pts,
               imbals,
               taus,
               trials,
               num_algs,
               num_agents,
               stream_size,
               num_test_pts,
               num_epochs,
               batch_size,
               num_workers,
               num_classes,
               device,
               train_path,
               val_path,
               num_sel_rnds,
               embed_batch_size,
               embed_dim,
               folder_to_class_file,
               test_label_file,
               weights_path)
        
    rare_acc=torch.zeros(len(trials),num_sel_rnds+1,num_algs)
    all_acc=torch.zeros(len(trials),num_sel_rnds+1,num_algs)
    
    sizes=torch.zeros(len(trials),num_sel_rnds+1,num_algs,num_classes)
    sum_sizes=torch.zeros(len(trials),num_sel_rnds+1,1)
    
    classes = random.sample(list(np.arange(1000)), num_classes)
    
    rare_classes = classes[:int(num_classes/2)]
    common_classes = classes[int(num_classes/2):]
    class_dict = dict(zip(rare_classes+common_classes, range(num_classes)))

    embeds, labels, idx_conv_dict = get_embeds(train_path,
                                               class_dict,
                                               embed_dim,
                                               embed_batch_size,
                                               num_classes,
                                               num_workers,
                                               weights_path,
                                               folder_to_class_file,
                                               device)
    
    test_embeds_loader, rare_val_embeds_loader, common_val_embeds_loader, val_embeds_loader = get_test_embed_loaders(embed_dim,
                                                                                                                     embed_batch_size,
                                                                                                                     batch_size,
                                                                                                                     num_classes,
                                                                                                                     num_workers,
                                                                                                                     weights_path,
                                                                                                                     val_path,
                                                                                                                     test_label_file,
                                                                                                                     class_dict,
                                                                                                                     num_test_pts,
                                                                                                                     device)

    init_x = torch.empty(0)
    init_y = torch.empty(0)

    stream_datasets_dict = {key: None for key in range(num_agents)}

    for agent in range(num_agents):
        agent_init_dataset, agent_stream_dataset = get_datasets(embeds, labels, imbals[agent], num_init_pts, num_classes)
        agent_init_loader = DataLoader(agent_init_dataset, batch_size=num_init_pts, num_workers=num_workers, shuffle=True)
        agent_init_samples = enumerate(agent_init_loader)
        _, (agent_init_x, agent_init_y) = next(agent_init_samples)

        init_x = torch.cat((init_x, agent_init_x[:int(np.ceil(num_init_pts/num_agents))]))
        init_y = torch.cat((init_y, agent_init_y[:int(np.ceil(num_init_pts/num_agents))]))

        stream_datasets_dict[agent] = agent_stream_dataset

    sizes[:,0] = (torch.stack((torch.tensor([(init_y==i).sum() for i in range(num_classes)]),
                               torch.tensor([(init_y==i).sum() for i in range(num_classes)]))))

    sum_sizes[:,0] = len(init_x)

    init_loader = DataLoader(TensorDataset(init_x, init_y), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = LogRegModel(embed_dim, num_classes) 
    model = train(device, num_epochs, init_loader, model)
        
    rare_isoreg = train_isoreg(model, rare_val_embeds_loader)
    common_isoreg = train_isoreg(model, common_val_embeds_loader)
    
    rare_acc[:,0] = (
            torch.cat((calc_acc(model, test_embeds_loader, num_classes)[0], 
                       calc_acc(model, test_embeds_loader, num_classes)[0])))
    
    all_acc[:,0] = (
            torch.cat((calc_acc(model, test_embeds_loader, num_classes)[1],
                       calc_acc(model, test_embeds_loader, num_classes)[1])))
    
    for trial in trials:
        FED_DMGT_model = load_model(model, embed_dim, num_classes, device)
        RAND_model = load_model(model, embed_dim, num_classes, device)

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
                tau = taus[agent]

                _, (agent_stream_x, agent_stream_y) = next(stream_samples_dict[agent])
                agent_FED_DMGT_x, agent_FED_DMGT_y, agent_RAND_x, agent_RAND_y = get_subsets(agent_stream_x,
                                                                                             agent_stream_y,
                                                                                             tau,
                                                                                             FED_DMGT_model,
                                                                                             num_classes,
                                                                                             is_isoreg,
                                                                                             rare_isoreg,
                                                                                             common_isoreg,
                                                                                             device)

                FED_DMGT_x = torch.cat((FED_DMGT_x, agent_FED_DMGT_x))
                FED_DMGT_y = torch.cat((FED_DMGT_y, agent_FED_DMGT_y))
                RAND_x = torch.cat((RAND_x, agent_RAND_x))
                RAND_y = torch.cat((RAND_y, agent_RAND_y))
            
            sizes[trial,sel_rnd+1] = (
                    sizes[trial,sel_rnd] + 
                    torch.stack((torch.tensor([(FED_DMGT_y==i).sum() for i in range(num_classes)]),
                                 torch.tensor([(RAND_y==i).sum() for i in range(num_classes)]))))

            sum_sizes[trial,sel_rnd+1] = (
                    torch.tensor([sum_sizes[trial,sel_rnd] + len(FED_DMGT_y)]))
                
            FED_DMGT_model = train(device,
                                   num_epochs,
                                   DataLoader(TensorDataset(FED_DMGT_x, FED_DMGT_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                                   FED_DMGT_model)

            rare_isoreg = train_isoreg(FED_DMGT_model, rare_val_embeds_loader)
            common_isoreg = train_isoreg(FED_DMGT_model, common_val_embeds_loader)
                
            RAND_model = train(device,
                               num_epochs,
                               DataLoader(TensorDataset(RAND_x, RAND_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                               RAND_model)
            
            rare_acc[trial,sel_rnd+1] = (
                    torch.cat((calc_acc(FED_DMGT_model, test_embeds_loader, num_classes)[0], 
                               calc_acc(RAND_model, test_embeds_loader, num_classes)[0])))
            
            all_acc[trial,sel_rnd+1] = (
                    torch.cat((calc_acc(FED_DMGT_model, test_embeds_loader, num_classes)[1],
                               calc_acc(RAND_model, test_embeds_loader, num_classes)[1])))
    
    df = fed_dmgt_df(rare_acc, all_acc, sizes, sum_sizes, trials, num_sel_rnds)
    
    return df

