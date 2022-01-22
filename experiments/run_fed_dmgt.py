# import libraries
import numpy as np
import random
import torch
import argparse
import sys
# import main experiment functions
from utils.main_exps.mnist_fed_dmgt import experiment as mnist_exp
from utils.main_exps.imnet_fed_dmgt import experiment as imnet_exp
sys.path.insert(0, '..')
from plots.fig5 import plot_figure5

parser = argparse.ArgumentParser()
# experiment parameters for mnist and imagenet
parser.add_argument('--dataset_name', type=str, default='imagenet')
parser.add_argument('--train_path', type=str, default='path/to/imagenet/train')
parser.add_argument('--val_path', type=str, default='path/to/imagenet/val')
parser.add_argument('--num_init_pts', type=int, default=1000)
parser.add_argument('--imbals', nargs='+', type=int, default=[2,5,10])
parser.add_argument('--taus', type=float, default=[0.1,0.1,0.1])
parser.add_argument('--num_agents', type=int, default=3)
parser.add_argument('--trials', nargs='+', type=int, default=np.arange(5))
parser.add_argument('--num_algs', type=int, default=2)
parser.add_argument('--stream_size', type=int, default=1000)
parser.add_argument('--num_test_pts', type=int, default=5000)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)

# extra imagenet parameters
parser.add_argument('--imnet_num_sel_rnds', type=int, default=6)
parser.add_argument('--imnet_embed_batch_size', type=int, default=256)
parser.add_argument('--imnet_embed_dim', type=int, default=2048)
parser.add_argument('--imnet_folder_to_class_file', type=str, default='../../imagenet_datafiles/folder_to_class.txt')
parser.add_argument('--imnet_test_label_file', type=str, default='../../imagenet_datafiles/test_classes.txt')
# can find simclr resnet50 model for pytorch here: https://github.com/tonylins/simclr-converter -- we use ResNet-50(1x)
parser.add_argument('--imnet_smclr_weights_path', type=str, default='path/to/simclr_resnet50/model') 

# extra mnist paramaters
parser.add_argument('--mnist_num_sel_rnds', type=int, default=10)

if __name__ == "__main__":
    
    args = parser.parse_args()
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 

    # fix randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed) 
    torch.backends.cudnn.deterministic=True
    
    input_args = [args.num_init_pts,
                  args.imbals,
                  args.taus,
                  args.trials,
                  args.num_agents,
                  args.num_algs,
                  args.stream_size,
                  args.num_test_pts,
                  args.num_epochs,
                  args.batch_size,
                  args.num_workers,
                  args.num_classes,
                  device]
    
    mnist_args = [args.train_path,
                  args.mnist_num_sel_rnds]

    imnet_args = [args.train_path,
                  args.val_path,
                  args.imnet_num_sel_rnds,
                  args.imnet_embed_batch_size,
                  args.imnet_embed_dim,
                  args.imnet_folder_to_class_file,
                  args.imnet_test_label_file,
                  args.imnet_smclr_weights_path]

    # generate dataframes from main experiment
    if args.dataset_name=='imagenet':
        df = imnet_exp(*input_args, *imnet_args)
        num_sel_rnds = args.imnet_num_sel_rnds
    else:
        df = mnist_exp(*input_args, *mnist_args)
        num_sel_rnds = args.mnist_num_sel_rnds

    ### Plot Figures ###

    fig_dir = '/save/figures/in/this/dir/'
    
    # figure 5
    plot_figure5(df, args.trials, np.arange(num_sel_rnds), args.dataset_name, fig_dir)

