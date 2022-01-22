# import libraries
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset
# import internal functions 
from ..exp_utils import class_card, get_subsets, MnistResNet, train, load_model, calc_acc, train_isoreg
from ..data_utils.mnist_data_utils import get_datasets, get_val_loaders
from ..dataframes import dmgt_df

# main experiment -- runs DMGT and RAND; generates all data for figures
def experiment(init_pts,
               imbals,
               taus,
               trials,
               num_algs,
               stream_size,
               num_test_pts,
               num_epochs,
               batch_size,
               num_workers,
               num_classes,
               device,
               data_path,
               num_sel_rnds):

    rare_acc=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,num_algs)
    all_acc=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,num_algs)
    
    sizes=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,num_algs,num_classes)
    sum_sizes=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,1)
    
    test_loader, rare_val_loader, common_val_loader, val_loader = get_val_loaders(num_test_pts, batch_size, num_workers, num_classes, data_path)

    for init_pts_idx,num_init_pts in enumerate(init_pts):    
        for imbal_idx,imbal in enumerate(imbals):
            model, stream_dataset, sizes, sum_sizes, rare_acc, all_acc = train_init_model(test_loader,
                                                                                          num_init_pts,
                                                                                          imbal,
                                                                                          init_pts_idx,
                                                                                          imbal_idx,
                                                                                          num_classes,
                                                                                          batch_size,
                                                                                          num_epochs,
                                                                                          num_workers,
                                                                                          sizes,
                                                                                          sum_sizes,
                                                                                          rare_acc,
                                                                                          all_acc,
                                                                                          device,
                                                                                          data_path)
            
            for tau_idx,tau in enumerate(taus):
                for trial in trials:
                    DMGT_model = load_model(model, device)
                    RAND_model = load_model(model, device)
            
                    stream_loader = DataLoader(stream_dataset, batch_size=stream_size, num_workers=num_workers, shuffle=True)
                    stream_samples = enumerate(stream_loader)
                    
                    for sel_round in range(num_sel_rnds):
                        _, (stream_x, stream_y) = next(stream_samples)
                        DMGT_model, RAND_model, sizes, sum_sizes, rare_acc, all_acc = update_models(DMGT_model,
                                                                                                    RAND_model,
                                                                                                    rare_val_loader,
                                                                                                    common_val_loader,
                                                                                                    test_loader,
                                                                                                    stream_x,
                                                                                                    stream_y,
                                                                                                    init_pts_idx,
                                                                                                    imbal_idx,
                                                                                                    tau_idx,
                                                                                                    tau,
                                                                                                    trial,
                                                                                                    sel_round,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    num_workers,
                                                                                                    num_classes,
                                                                                                    sizes,
                                                                                                    sum_sizes,
                                                                                                    rare_acc,
                                                                                                    all_acc,
                                                                                                    device)
        
    df = dmgt_df(rare_acc, all_acc, sizes, sum_sizes, init_pts, imbals, taus, trials, num_sel_rnds)
    
    return df

######## Internal Functions ########

# warm-start trains DMGT and RAND models
def train_init_model(test_loader,
                     num_init_pts,
                     imbal,
                     init_pts_idx,
                     imbal_idx,
                     num_classes,
                     batch_size,
                     num_epochs,
                     num_workers,
                     sizes,
                     sum_sizes,
                     rare_acc,
                     all_acc,
                     device,
                     data_path):

    init_dataset, stream_dataset = get_datasets(num_init_pts, imbal, num_classes, data_path)
    init_loader = DataLoader(init_dataset, batch_size=num_init_pts, num_workers=num_workers, shuffle=True)
    init_samples = enumerate(init_loader)
    _, (init_x, init_y) = next(init_samples)
    
    sizes[:,:,:,:,0] = (
            torch.stack((torch.tensor([(init_y==i).float().sum() for i in range(num_classes)]),
                         torch.tensor([(init_y==i).float().sum() for i in range(num_classes)]))))
    
    sum_sizes[:,:,:,:,0] = num_init_pts
    
    init_loader = DataLoader(TensorDataset(init_x, init_y), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    model = MnistResNet()
    model = train(device, num_epochs, init_loader, model)

    rare_acc[:,:,:,:,0] = (
            torch.cat((calc_acc(model, test_loader, num_classes)[0], 
                       calc_acc(model, test_loader, num_classes)[0])))
    
    all_acc[:,:,:,:,0] = (
            torch.cat((calc_acc(model, test_loader, num_classes)[1],
                       calc_acc(model, test_loader, num_classes)[1])))
    
    return model, stream_dataset, sizes, sum_sizes, rare_acc, all_acc

# updates models on selected points after each batch
def update_models(DMGT_model,
                  RAND_model,
                  rare_val_loader,
                  common_val_loader,
                  test_loader,
                  stream_x,
                  stream_y,
                  init_pts_idx,
                  imbal_idx,
                  tau_idx,
                  tau,
                  trial,
                  sel_round,
                  num_epochs,
                  batch_size,
                  num_workers,
                  num_classes,
                  sizes,
                  sum_sizes,
                  rare_acc,
                  all_acc,
                  device):

    rare_isoreg = train_isoreg(DMGT_model, rare_val_loader)
    common_isoreg = train_isoreg(DMGT_model, common_val_loader)
    
    DMGT_x, DMGT_y, RAND_x, RAND_y = get_subsets(stream_x, stream_y, tau, DMGT_model, num_classes, rare_isoreg, common_isoreg, device)
    
    sizes[init_pts_idx,imbal_idx,tau_idx,trial,sel_round+1] = (
            torch.stack((torch.tensor([(DMGT_y==i).float().sum() for i in range(num_classes)]),
                         torch.tensor([(RAND_y==i).float().sum() for i in range(num_classes)]))))
    
    sum_sizes[init_pts_idx,imbal_idx,tau_idx,trial,sel_round+1] = (
            torch.tensor([sum_sizes[init_pts_idx,imbal_idx,tau_idx,trial,sel_round] + len(DMGT_y)]))
        
    DMGT_model = train(device,
                       num_epochs,
                       DataLoader(TensorDataset(DMGT_x, DMGT_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                       DMGT_model)
    
    RAND_model = train(device,
                       num_epochs,
                       DataLoader(TensorDataset(RAND_x, RAND_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                       RAND_model)
    
    rare_acc[init_pts_idx,imbal_idx,tau_idx,trial,sel_round+1] = (
            torch.cat((calc_acc(DMGT_model, test_loader, num_classes)[0], 
                       calc_acc(RAND_model, test_loader, num_classes)[0])))
    
    all_acc[init_pts_idx,imbal_idx,tau_idx,trial,sel_round+1] = (
            torch.cat((calc_acc(DMGT_model, test_loader, num_classes)[1],
                       calc_acc(RAND_model, test_loader, num_classes)[1])))
    
    return DMGT_model, RAND_model, sizes, sum_sizes, rare_acc, all_acc

