# import libraries
import numpy as np
import sympy
from sympy.solvers import solve 
from sympy import Symbol 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse

def plot_figure4(df,
                 num_init_pts,
                 imbal,
                 tau,
                 trials,
                 sel_rnds,
                 num_classes,
                 dataset_name,
                 fig_dir):
    
    n = Symbol('n')
    exp_lab_size = int((num_classes/2)*np.ceil(solve(sympy.sqrt(n+1) - sympy.sqrt(n) - tau, n)[0]))
    
    fig, ax = plt.subplots()
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    df = (df[(df['sel_rnd'].isin(sel_rnds)) &
             (df['num_init_pts']==num_init_pts) &
             (df['imbal']==imbal) &
             (df['tau']==tau)][['sel_rnd',
                                'DMGT_rare_amnt',
                                'RAND_rare_amnt',
                                'DMGT_common_amnt',
                                'RAND_common_amnt']])
    
    plot = sns.lineplot(data=df[['sel_rnd', 'DMGT_common_amnt']],
                        x='sel_rnd',
                        y='DMGT_common_amnt',
                        color=sns.color_palette('muted')[0],
                        linewidth=2,
                        ci=95,
                        estimator='mean')
    
    sns.lineplot(data=df[['sel_rnd', 'DMGT_rare_amnt']],
                 x='sel_rnd',
                 y='DMGT_rare_amnt',
                 linewidth=2,
                 color=sns.color_palette('muted')[1],
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(data=df[['sel_rnd', 'RAND_common_amnt']],
                 x='sel_rnd',
                 y='RAND_common_amnt',
                 color=sns.color_palette('muted')[2],
                 linewidth=2,
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(data=df[['sel_rnd', 'RAND_rare_amnt']],
                 x='sel_rnd',
                 y='RAND_rare_amnt',
                 linewidth=2,
                 color=sns.color_palette('muted')[3],
                 ci=95,
                 estimator='mean')

    
    sns.lineplot(x=sel_rnds,
                 y=exp_lab_size*np.ones(len(sel_rnds)),
                 linewidth=2,
                 color=sns.color_palette('muted')[4])
     
    round_skip=1 if dataset_name=='imagenet' else 2
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(sel_rnds[::round_skip]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{i}' for i in sel_rnds[::round_skip]]))
    
    plot.set_xlabel('Round', fontsize=14, labelpad=7)
    plot.set_ylabel('Size of Selected Set', fontsize=14, labelpad=7)
    
    fig.subplots_adjust(
        top=0.75,
        bottom=0.1,
        left=0.13,
        right=0.97
    )

    fig.legend(labels=['DMGT: common classes',
                       'DMGT: rare classes',
                       'RAND: common classes',
                       'RAND: rare classes',
                       'Ideal rare/common amounts'],
               ncol=1,
               loc='upper left')

    fig.savefig(fig_dir + f'{dataset_name}_class_balance.pdf')

