# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse

def plot_figure5(df,
                 trials,
                 sel_rnds,
                 stream_size,
                 dataset_name)

    fig, ax = plt.subplots() 
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plot=sns.lineplot(ax=ax,
                      data=df[['sel_rnd', 'FED_DMGT_all_acc']],
                      x='sel_rnd',
                      y='FED_DMGT_all_acc',
                      color=sns.color_palette('pastel')[0],
                      ci=95,
                      estimator='mean')
    
    sns.lineplot(ax=ax,
                 data=df[['sel_rnd', 'RAND_all_acc']],
                 x='sel_rnd',
                 y='RAND_all_acc',
                 color=sns.color_palette('pastel')[2],
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(ax=ax,
                 data=df[['sel_rnd', 'FED_DMGT_rare_acc']],
                 x='sel_rnd',
                 y='FED_DMGT_rare_acc',
                 color=sns.color_palette('muted')[0],
                 linestyle='--',
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(ax=ax,
                 data=df[['sel_rnd', 'RAND_rare_acc']],
                 x='sel_rnd',
                 y='RAND_rare_acc',
                 color=sns.color_palette('muted')[2],
                 linestyle='--',
                 ci=95,
                 estimator='mean')
    
    sizes_labels = []
    for sel_rnd in sel_rnds:
        sizes_labels += [int(np.floor(df[df['sel_rnd']==sel_rnd]['sum_sizes'].mean()))]
    
    round_skip=2 if dataset_name=='MNIST' else 1
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(sel_rnds[::round_skip]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{i}'+'\n'+f'{sizes_labels[i]}' for i in sel_rnds[::round_skip]]))
    
    plot.set_xlabel(None)
    plot.set_ylabel('Accuracy', fontsize=14, labelpad=7)
    
    ax.legend(labels=['FED_DMGT: all classes',
                      'RAND: all classes',
                      'FED_DMGT: rare classes',
                      'RAND: rare classes'],
              loc='lower right')

    fig.text(0.001, 0.04, 'Round\nAvg. # sel. pts', fontsize=8)
    fig.tight_layout()
    #fig.savefig(save_dir + f'_{dataset_name}_accuracy.pdf')

