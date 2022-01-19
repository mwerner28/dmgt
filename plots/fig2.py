import numpy as np
import sympy
from sympy.solvers import solve 
from sympy import Symbol 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse
from ../experiments/run_exp import imnet_dmgt_df

def plot_figure2(df,
                 num_init_pts,
                 imbal,
                 trial,
                 taus,
                 sel_rnd,
                 num_classes,
                 save_dir,
                 dataset_name):
    
    fig, ax = plt.subplots()
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    df = (df[(df['sel_rnd']==sel_rnd) &
             (df['num_init_pts']==num_init_pts) &
             (df['imbal']==imbal)][['tau',
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
    
    fig.savefig(f'{dataset_name}_tau_v_size.pdf', bbox_inches = "tight")

parser.add_argument('--num_init_pts', type=int, default=1000)
parser.add_argument('--imbal', type=int, default=5)
parser.add_argument('--taus', nargs='+', type=float, default=np.arange(0.05,0.45,0.05))
parser.add_argument('--trials', nargs='+', type=int, default=np.arange(5))
parser.add_argumen('--num_sel_rounds', type=int, default=6)
parser.add_argument('--stream_size', type=int, default=1000)
parser.add_argument('--num_test_pts', type=int, default=5000)
parser.add_argument('--is_isoreg', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)

if __name__=='main':
    
    args = parser.parse_args()
    save_dir = SAVE PLOTS HERE

    # figure 2
    plot_figure2(imnet_dmgt_df,
                 args.num_init_pts,
                 args.imbal,
                 args.taus,
                 args.trials[0],
                 args.num_sel_rnds,
                 args.num_classes,
                 save_dir,
                 dataset_name='IMNET')
