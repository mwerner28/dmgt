import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse

def plot_figure3(df,
                 num_init_pts,
                 imbal,
                 tau,
                 trials,
                 sel_rnds,
                 dataset_name,
                 fig_dir):
    
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False) 
    
    df = (df[(df['sel_rnd'].isin(sel_rnds)) &
             (df['num_init_pts']==num_init_pts) &
             (df['imbal']==imbal) &
             (df['tau']==tau)][['sel_rnd',
                                'DMGT_all_acc',
                                'RAND_all_acc',
                                'DMGT_rare_acc',
                                'RAND_rare_acc',
                                'sum_sizes']])
                 
    plot=sns.lineplot(data=df[['sel_rnd', 'DMGT_all_acc']],
                      x='sel_rnd',
                      y='DMGT_all_acc',
                      color=sns.color_palette('pastel')[0],
                      ci=95,
                      estimator='mean')
    
    sns.lineplot(data=df[['sel_rnd', 'RAND_all_acc']],
                 x='sel_rnd',
                 y='RAND_all_acc',
                 color=sns.color_palette('pastel')[2],
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(data=df[['sel_rnd', 'DMGT_rare_acc']],
                 x='sel_rnd',
                 y='DMGT_rare_acc',
                 color=sns.color_palette('muted')[0],
                 linestyle='--',
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(data=df[['sel_rnd', 'RAND_rare_acc']],
                 x='sel_rnd',
                 y='RAND_rare_acc',
                 color=sns.color_palette('muted')[2],
                 linestyle='--',
                 ci=95,
                 estimator='mean')
    
    sizes_labels = []
    for sel_rnd in sel_rnds:
        sizes_labels += [int(np.floor(df[df['sel_rnd']==sel_rnd]['sum_sizes'].mean()))]
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(sel_rnds[::round_skip]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{i}'+'\n'+f'{sizes_labels[i]}' for i in sel_rnds[::round_skip]]))
   
    round_skip=1 if dataset_name=='imagenet' else 2
    
    plot.set_xlabel(None)
    plot.set_ylabel('Accuracy', fontsize=14, labelpad=7)
    
    ax.legend(labels=['DMGT: all classes',
                      'RAND: all classes',
                      'DMGT: rare classes',
                      'RAND: rare classes'],
              loc='lower right')
    
    fig.text(0.001, 0.04, 'Round\nAvg. # sel. pts', fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir + f'{dataset_name}_accuracy.pdf')

