# import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse
# import dataframes from main experiment on mnist and imagenet
from ../experiments/dmgt/run_exp import mnist_df, imnet_df

def plot_figure3(df,
                 num_init_pts,
                 imbal,
                 tau,
                 trials,
                 sel_rnds,
                 stream_size,
                 num_test_pts,
                 save_dir,
                 dataset_name):
    
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
                                  'sum_sizes',
                                  'trial']])
                 
    plot=sns.lineplot(ax=ax,
                      data=df[df['trial'].isin(trials)][['sel_rnd', 'DMGT_all_acc']],
                      x='sel_rnd',
                      y='DMGT_all_acc',
                      color=sns.color_palette('pastel')[0],
                      ci=95,
                      estimator='mean')
    
    sns.lineplot(ax=ax,
                 data=df[df['trial'].isin(trials)][['sel_rnd', 'RAND_all_acc']],
                 x='sel_rnd',
                 y='RAND_all_acc',
                 color=sns.color_palette('pastel')[2],
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(ax=ax,
                 data=df[df['trial'].isin(trials)][['sel_rnd', 'DMGT_rare_acc']],
                 x='sel_rnd',
                 y='DMGT_rare_acc',
                 color=sns.color_palette('muted')[0],
                 linestyle='--',
                 ci=95,
                 estimator='mean')
    
    sns.lineplot(ax=ax,
                 data=df[df['trial'].isin(trials)][['sel_rnd', 'RAND_rare_acc']],
                 x='sel_rnd',
                 y='RAND_rare_acc',
                 color=sns.color_palette('muted')[2],
                 linestyle='--',
                 ci=95,
                 estimator='mean')
    
    sizes_labels = []
    for sel_rnd in sel_rnds:
        sizes_labels += [int(np.floor(df[df['sel_rnd']==sel_rnd]['sum_sizes'].mean()))]
    
    if dataset_name=='MNIST':
        round_skip=2

        plot.set_ylabel('Accuracy', fontsize=14, labelpad=7)
        fig.text(0.001, 0.04, 'Round\nAvg. # sel. pts', fontsize=8)
    
    else:
        round_skip=1

        ax.legend(labels=['DMGT: all classes',
                          'RAND: all classes',
                          'DMGT: rare classes',
                          'RAND: rare classes'],
                  loc='lower right')
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(sel_rnds[::round_skip]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{i}'+'\n'+f'{sizes_labels[i]}' for i in sel_rnds[::round_skip]]))
    
    plot.set_xlabel(None)
    
    fig.tight_layout()
    fig.savefig(save_dir + f'_{dataset_name}_accuracy.pdf')

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

    # figure 3a
    plot_figure3(mnist_df,
                 args.num_init_pts,
                 args.imbal,
                 args.tau,
                 args.trials,
                 np.arange(args.mnist_num_sel_rnds),
                 args.stream_size,
                 args.num_test_pts,
                 save_dir,
                 dataset_name='MNIST')

    # figure 3b
    plot_figure3(imnet_df,
                 args.num_init_pts,
                 args.imbal,
                 args.tau,
                 args.trials,
                 np.arange(imnet_num_sel_rnds),
                 args.stream_size,
                 args.num_test_pts,
                 save_dir,
                 dataset_name='IMNET')
