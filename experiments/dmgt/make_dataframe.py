import torch
import pandas as pd

def dmgt_df(data, init_pts, imbals, taus, trials, num_sel_rnds):

    rare_acc, all_acc, sizes, sum_sizes = data
    
    df = pd.DataFrame(columns=['num_init_pts',
                               'imbal',
                               'tau',
                               'trial',
                               'sel_rnd',
                               'DMGT_all_acc',
                               'RAND_all_acc'
                               'DMGT_rare_acc',
                               'RAND_rare_acc',
                               'DMGT_rare_amnt',
                               'RAND_rare_amnt',
                               'DMGT_common_amnt',
                               'RAND_common_amnt',
                               'sum_sizes'])
    
    for init_pts_idx, num_init_pts in enumerate(init_pts):
        for imbal_idx, imbal in enumerate(imbals):
            for tau_idx, tau in enumerate(taus):
                for trial in trials:
                    df = df.append(pd.DataFrame({'num_init_pts':num_init_pts*torch.ones(num_sel_rnds+1),
                                                 'imbal':imbal*torch.ones(num_sel_rnds+1),
                                                 'tau':tau*torch.ones(num_sel_rnds+1),
                                                 'trial':trial*torch.ones(num_sel_rnds+1),
                                                 'sel_rnd':torch.arange(num_sel_rnds+1),
                                                 'DMGT_all_acc':all_acc[init_pts_idx,imbal_idx,tau_idx,trial,:,0].squeeze(),
                                                 'RAND_all_acc':all_acc[init_pts_idx,imbal_idx,tau_idx,trial,:,1].squeeze(),
                                                 'DMGT_rare_acc':rare_acc[init_pts_idx,imbal_idx,tau_idx,trial,:,0].squeeze(),
                                                 'RAND_rare_acc':rare_acc[init_pts_idx,imbal_idx,tau_idx,trial,:,1].squeeze(),
                                                 'DMGT_rare_amnt':(torch.stack([x[:5].sum().int()
                                                                 for x in sizes[init_pts_idx,imbal_idx,tau_idx,trial,:,0]])),
                                                 'RAND_rare_amnt':(torch.stack([x[:5].sum().int()
                                                                   for x in sizes[init_pts_idx,imbal_idx,tau_idx,trial,:,1]])),
                                                 'DMGT_common_amnt':(torch.stack([x[5:].sum().int()
                                                                   for x in sizes[init_pts_idx,imbal_idx,tau_idx,trial,:,0]])),
                                                 'RAND_common_amnt':(torch.stack([x[5:].sum().int()
                                                                     for x in sizes[init_pts_idx,imbal_idx,tau_idx,trial,:,1]])),
                                                 'sum_sizes':sum_sizes[init_pts_idx,imbal_idx,tau_idx,trial,:].squeeze().int()}),
                                                 ignore_index=True)
    
    return df

