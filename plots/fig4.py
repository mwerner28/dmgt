import numpy as np
import sympy
from sympy.solvers import solve 
from sympy import Symbol 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse
from ../experiments/dmgt/run_exp import mnist_df, imnet_df

def plot_figure4(df,
                 num_init_pts,
                 imbal,
                 tau,
                 trials,
                 sel_rnds,
                 stream_size,
                 num_test_pts,
                 num_classes,
                 save_dir,
                 dataset_name):
    
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
                                  'RAND_common_amnt',
                                  'trial']])
    
    plot = sns.lineplot(data=df[df['trial'].isin(trials)][['sel_rnd', 'DMGT_common_amnt']],
                        x='sel_rnd',
                        y='DMGT_common_amnt',
                        color=sns.color_palette('muted')[0],
                        linewidth=2,
                        ci=95,
                        estimator='mean')
    
    sns.lineplot(data=df[df['trial'].isin(trials)][['sel_rnd', 'DMGT_rare_amnt']],
                 x='sel_rnd',
                 y='DMGT_rare_amnt',
                 linewidth=2,
                 color=sns.color_palette('muted')[1],
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(data=df[df['trial'].isin(trials)][['sel_rnd', 'RAND_common_amnt']],
                 x='sel_rnd',
                 y='RAND_common_amnt',
                 color=sns.color_palette('muted')[2],
                 linewidth=2,
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(data=df[df['trial'].isin(trials)][['sel_rnd', 'RAND_rare_amnt']],
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
     
    round_skip=2 if dataset_name=='MNIST' else 1
    
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

    fig.savefig(save_dir + f'_{dataset_name}_class_balance.pdf')

parser = argparse.ArgumentParser()
parser.add_argument('--num_init_pts', type=int, default=1000)
parser.add_argument('--imbal', type=int, default=5)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--trials', nargs='+', type=int, default=np.arange(5))
parser.add_argument('--mnist_num_sel_rnds', type=int, default=15)
parser.add_argument('--imnet_num_sel_rnds', type=int, default=6)
parser.add_argument('--stream_size', type=int, default=1000)
parser.add_argument('--num_test_pts', type=int, default=5000)
parser.add_argument('--seed', type=int, default=1)

if __name__=='main':
    
    args = parser.parse_args()
    save_dir = SAVE PLOTS HERE

    # plot figure 4a
    plot_figure4(mnist_df,
                 args.num_init_pts,
                 args.imbal,
                 args.tau,
                 args.trials,
                 np.arange(args.mnist_num_sel_rnds),
                 args.stream_size,
                 args.num_test_pts,
                 args.num_classes,
                 save_dir,
                 dataset_name='MNIST')

    # plot figure 4b
    plot_figure4(imnet_df,
                 args.num_init_pts,
                 args.imbal,
                 args.tau,
                 args.trials,
                 np.arange(args.imnet_num_sel_rnds),
                 args.stream_size,
                 args.num_test_pts,
                 args.num_classes,
                 save_dir,
                 dataset_name='IMNET')
