# import libraries
import numpy as np
import argparse
# import main experiment functions
from mnist import experiment as mnist_exp
from imnet import experiment as imnet_exp

parser = argparse.ArgumentParser()
# experiment parameters for mnist and imagenet
parser.add_argument('--num_init_pts', type=int, default=1000)
parser.add_argument('--imbals', nargs='+', type=int, default=[2,5,10])
parser.add_argument('--taus', type=float, default=[0.1,0.1,0.1])
parser.add_argument('--num_agents', type=int, default=3)
parser.add_argument('--trials', nargs='+', type=int, default=np.arange(5))
parser.add_argument('--mnist_num_sel_rnds', type=int, default=10)
parser.add_argument('--imnet_num_sel_rnds', type=int, default=6)
parser.add_argument('--num_algs', type=int, default=2)
parser.add_argument('--stream_size', type=int, default=1000)
parser.add_argument('--num_test_pts', type=int, default=5000)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)

# extra mnist paramaters
parser.add_argument('--mnist_train_dir', type=str, default='path/to/mnist/train/')
parser.add_argument('--mnist_val_dir', type=str, default='path/to/mnist/val/')

# extra imagenet parameters
parser.add_argument('--imnet_embed_dim', type=int, default=2048)
parser.add_argument('--imnet_train_dir', type=str, default='path/to/imagenet/train/')
parser.add_argument('--imnet_val_dir', type=str, default='path/to/imagenet/val/')
parser.add_argument('--imnet_folder_to_class_file', type=str, default='../../imagenet_datafiles/folder_to_class.txt')
parser.add_argument('--imnet_test_label_file', type=str, default='../../imagenet_datafiles/test_classes.txt')
# can find simclr resnet50 model for pytorch here: https://github.com/tonylins/simclr-converter -- we use ResNet-50(1x)
parser.add_argument('--imnet_smclr_weights_path', type=str, default='path/to/simclr_resnet50/model') 

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
                  args.tau,
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
    
    mnist_args = [args.mnist_num_sel_rnds,
                  args.mnist_train_dir,
                  args.mnist_val_dir]

    imnet_args = [args.imnet_num_sel_rnds,
                  args.imnet_embed_batch_size,
                  args.imnet_embed_dim,
                  args.imnet_train_dir,
                  args.imnet_val_dir,
                  args.imnet_folder_to_class_file,
                  args.imnet_test_label_file,
                  args.imnet_smclr_weights_path]

    # generate dataframes from main experiment
    mnist_df = mnist_exp(*input_args, *mnist_args)
    
    imnet_df = imnet_exp(*input_args, *imnet_args)
    
    return mnist_df, imnet_df
