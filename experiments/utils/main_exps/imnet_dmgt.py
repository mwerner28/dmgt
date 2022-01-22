import numpy as np 
import random
import torch 
from torch.utils.data import DataLoader, TensorDataset
# import internal functions
from ..exp_utils import class_card, get_subsets, Embed, LogRegModel, train, load_model, calc_acc, train_isoreg
from ..data_utils.imnet_data_utils import get_embed_loader, get_embeds, get_test_embed_loaders, get_test_loader
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
               train_path,
               val_path,
               num_sel_rnds,
               embed_batch_size,
               embed_dim,
               folder_to_class_file,
               test_label_file,
               weights_path):
    
    rare_acc=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,num_algs)
    all_acc=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,num_algs)
    
    sizes=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,num_algs,num_classes)
    sum_sizes=torch.zeros(len(init_pts),len(imbals),len(taus),len(trials),num_sel_rnds+1,1)
    
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
    for init_pts_idx, num_init_pts in enumerate(init_pts):
        for imbal_idx, imbal in enumerate(imbals):
            
            model, stream_dataset, sizes, sum_sizes, rare_acc, all_acc = train_init_model(test_embeds_loader,
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
                                                                                          embed_dim,
                                                                                          class_dict,
                                                                                          embeds,
                                                                                          labels,
                                                                                          device)
            
            
            
            for tau_idx, tau in enumerate(taus):
                for trial in trials:
                    
                    DMGT_model = load_model(model, device, embed_dim, num_classes)
                    RAND_model = load_model(model, device, embed_dim, num_classes)

                    stream_loader = DataLoader(stream_dataset, batch_size=stream_size, num_workers=num_workers, shuffle=True)
                    stream_samples = enumerate(stream_loader)
                    
                    for sel_rnd in range(num_sel_rnds):
                        _, (stream_x, stream_y) = next(stream_samples)
                        
                        DMGT_model, RAND_model, sizes, sum_sizes, rare_acc, all_acc = update_models(DMGT_model,
                                                                                                    RAND_model,
                                                                                                    rare_val_embeds_loader,
                                                                                                    common_val_embeds_loader,
                                                                                                    test_embeds_loader,
                                                                                                    stream_x,
                                                                                                    stream_y,
                                                                                                    init_pts_idx,
                                                                                                    imbal_idx,
                                                                                                    tau_idx,
                                                                                                    tau,
                                                                                                    trial,
                                                                                                    sel_rnd,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    num_workers,
                                                                                                    num_classes,
                                                                                                    sizes,
                                                                                                    sum_sizes,
                                                                                                    rare_acc,
                                                                                                    all_acc,
                                                                                                    class_dict,
                                                                                                    device)
    
    df = dmgt_df(rare_acc, all_acc, sizes, sum_sizes, init_pts, imbals, taus, trials, num_sel_rnds)                         
    
    return df

######## Internal Functions ########

# warm-start trains DMGT and RAND models
def train_init_model(test_embeds_loader,
                     num_init_pts,
                     imbal,
                     init_pts_idxs,
                     imbal_idx,
                     num_classes,
                     batch_size,
                     num_epochs,
                     num_workers,
                     sizes,
                     sum_sizes,
                     rare_acc,
                     all_acc,
                     embed_dim,
                     class_dict,
                     embeds,
                     labels,
                     device):

    init_dataset, stream_dataset = get_datasets(embeds, labels, num_init_pts, imbal, num_classes)

    init_loader = DataLoader(init_dataset, batch_size=num_init_pts, num_workers=num_workers, shuffle=True)

    init_samples = enumerate(init_loader)
    _, (init_x, init_y) = next(init_samples)

    sizes[:,:,:,:,0] = (

            torch.stack((torch.tensor([(init_y==i).sum() for i in range(num_classes)]),
                         torch.tensor([(init_y==i).sum() for i in range(num_classes)]))))

    sum_sizes[:,:,:,:,0] = num_init_pts

    init_loader = DataLoader(TensorDataset(init_x, init_y), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = LogRegModel(embed_dim, num_classes) 
    model = train(device, num_epochs, init_loader, model)

    rare_acc[:,:,:,:,0] = (
            torch.cat((calc_acc(model, test_embeds_loader, num_classes)[0],
                       calc_acc(model, test_embeds_loader, num_classes)[0])))

    all_acc[:,:,:,:,0] = (
            torch.cat((calc_acc(model, test_embeds_loader, num_classes)[1],
                       calc_acc(model, test_embeds_loader, num_classes)[1])))

    return model, stream_dataset, sizes, sum_sizes, rare_acc, all_acc

# updates models on DMGT-selected points after each batch
def update_models(DMGT_model,
                  RAND_model,
                  rare_val_embeds_loader,
                  common_val_embeds_loader,
                  test_embeds_loader,
                  stream_x,
                  stream_y,
                  init_pts_idx,
                  imbal_idx,
                  tau_idx,
                  tau,
                  trial,
                  sel_rnd,
                  num_epochs,
                  batch_size,
                  num_workers,
                  num_classes,
                  sizes,
                  sum_sizes,
                  rare_acc,
                  all_acc,
                  class_dict,
                  device):

    rare_isoreg = train_isoreg(DMGT_model, rare_val_embeds_loader)
    common_isoreg = train_isoreg(DMGT_model, common_val_embeds_loader)

    DMGT_x, DMGT_y, RAND_x, RAND_y = get_subsets(stream_x, stream_y, tau, DMGT_model, num_classes, rare_isoreg, common_isoreg, device)

    sizes[init_pts_idx,imbal_idx,tau_idx,trial,sel_rnd+1] = (
            torch.stack((torch.tensor([(DMGT_y==i).sum() for i in range(num_classes)]),
                         torch.tensor([(RAND_y==i).sum() for i in range(num_classes)]))))
    
    sum_sizes[init_pts_idx,imbal_idx,tau_idx,trial,sel_rnd+1] = (
            torch.tensor([sum_sizes[init_pts_idx,imbal_idx,tau_idx,trial,sel_rnd] + len(DMGT_y)]))
        
    DMGT_model = train(device,
                       num_epochs,
                       DataLoader(TensorDataset(DMGT_x, DMGT_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                       DMGT_model)

    RAND_model = train(device,
                       num_epochs,
                       DataLoader(TensorDataset(RAND_x, RAND_y), batch_size=batch_size, num_workers=num_workers, shuffle=True),
                       RAND_model)
    
    rare_acc[init_pts_idx,imbal_idx,tau_idx,trial,sel_rnd+1] = (
            torch.cat((calc_acc(DMGT_model, test_embeds_loader, num_classes)[0], 
                       calc_acc(RAND_model, test_embeds_loader, num_classes)[0])))
    
    all_acc[init_pts_idx,imbal_idx,tau_idx,trial,sel_rnd+1] = (
            torch.cat((calc_acc(DMGT_model, test_embeds_loader, num_classes)[1],
                       calc_acc(RAND_model, test_embeds_loader, num_classes)[1])))

    return DMGT_model, RAND_model, sizes, sum_sizes, rare_acc, all_acc


