import numpy as np
import sympy
from sympy.solvers import solve 
from sympy import Symbol 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse

def plot_figure2(df,
                 num_init_pts,
                 imbal,
                 trial,
                 taus,
                 sel_rnd,
                 num_classes,
                 dataset_name,
                 fig_dir):
    
    print ('got in here')
    fig, ax = plt.subplots()
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    df = (df[(df['sel_rnd']==sel_rnd) &
             (df['num_init_pts']==num_init_pts) &
             (df['imbal']==imbal) &
             (df['trial']==trial)][['tau',
                                    'DMGT_rare_amnt',
                                    'DMGT_common_amnt']])
    
    n = Symbol('n')
    
    ideal_sizes = np.asarray(list(map(lambda tau: int((num_classes/2)*np.ceil(solve(sympy.sqrt(n+1) - sympy.sqrt(n) - tau, n)[0])), taus)))
    
    plot = sns.lineplot(data=df,
                        x='tau',
                        y='DMGT_common_amnt',
                        ci=95,
                        color=sns.color_palette('pastel')[0],
                        estimator='mean',
                        linewidth=5)
    
    sns.lineplot(data=df,
                 x='tau',
                 y='DMGT_rare_amnt',
                 ci=95,
                 color=sns.color_palette('pastel')[3],
                 estimator='mean',
                 linewidth=5)
    
    sns.lineplot(x=taus,
                 y=ideal_sizes,
                 color='darkgrey',
                 linestyle='--',
                 linewidth=5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plot.set_xlabel('Threshold', fontsize=14)
    plot.set_ylabel('Size of Selected Set', fontsize=14)
    plot.legend(labels=['common','rare', 'ideal'], bbox_to_anchor=(0.7,0.3), prop={'size':8})
    fig.savefig(fig_dir + f'{dataset_name}_tau_v_size.pdf', bbox_inches = "tight")

