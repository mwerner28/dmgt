# import libraries
import torch
import pandas as pd

# creates dataframe from main experiment data
def fed_dmgt_df(data, trials, num_sel_rnds):

    rare_acc, all_acc, sizes, sum_sizes = data
    
    df = pd.DataFrame(columns=['trial',
                               'sel_rnd',
                               'FED_DMGT_all_acc',
                               'RAND_all_acc'
                               'FED_DMGT_rare_acc',
                               'RAND_rare_acc',
                               'FED_DMGT_rare_amnt',
                               'RAND_rare_amnt',
                               'FED_DMGT_common_amnt',
                               'RAND_common_amnt',
                               'sum_sizes'])
    
    for trial in trials:
        df = df.append(pd.DataFrame({'trial':trial*torch.ones(num_sel_rnds+1),
                                     'sel_rnd':torch.arange(num_sel_rnds+1),
                                     'FED_DMGT_all_acc':all_acc[trial,:,0].squeeze(),
                                     'RAND_all_acc':all_acc[trial,:,1].squeeze(),
                                     'FED_DMGT_rare_acc':rare_acc[trial,:,0].squeeze(),
                                     'RAND_rare_acc':rare_acc[trial,:,1].squeeze(),
                                     'FED_DMGT_rare_amnt':(torch.stack([x[:5].sum().int() for x in sizes[trial,:,0]])),
                                     'RAND_rare_amnt':(torch.stack([x[:5].sum().int() for x in sizes[trial,:,1]])),
                                     'FED_DMGT_common_amnt':(torch.stack([x[5:].sum().int() for x in sizes[trial,:,0]])),
                                     'RAND_common_amnt':(torch.stack([x[5:].sum().int() for x in sizes[trial,:,1]])),
                                     'sum_sizes':sum_sizes[trial].squeeze().int()}),
                                     ignore_index=True)
    
    return df

