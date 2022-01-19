import numpy as 
import argparse

from mnist import experiment as mnist_exp
from imnet import experiment as imnet_exp
from make_dataframe import fed_dmgt_df

parser = argparse.ArgumentParser()
parser.add_argument('--num_init_pts', type=int, default=1000)
parser.add_argument('--imbals', nargs='+', type=int, default=[2,5,10])
parser.add_argument('--tau', type=float, default=0.1)
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

## extra imagenet args
parser.add_argument('--embed_batch_size', type=int, default=256)
parser.add_argument('--embed_dim', type=int, default=2048)
parser.add_argument('--data_dir', type=str, default='/home/eecs/mwerner/imagenet/train/')
parser.add_argument('--test_dir', type=str, default='/work/data/imagenet/val/')
parser.add_argument('--folder_to_class_file', type=str, default='/home/eecs/mwerner/IMNET_CODE/map_clsloc.txt')
parser.add_argument('--test_label_file', type=str, default='/home/eecs/mwerner/IMNET_CODE/test_labels.txt')

### Imnet file paths and directories #####
# directory where ResNet50 model weights learned from SimCLR unsupervised training  
weights_dir = f'/home/eecs/mwerner/{args.dataset_name}_CODE/weights_dir/'

# SimCLR ResNet50 model weights .pkl file
weights_path=weights_dir + 'resnet50-1x.pth'

# directory where SimCLR embeddings are stored
embeds_dir = DIR WHERE SIMCLR EMBEDDINGS ARE STORED

# train and test SimCLR embeddings .pkl files
embeds_path=embeds_dir + f'embeds_1x_{args.embed_dim}.pkl'
embeds_labels_path=embeds_dir + 'labels_1x.pkl'
test_embeds_path=embeds_dir + f'test_embeds_1x_{args.embed_dim}.pkl'
test_embeds_labels_path=embeds_dir + f'test_labels_1x.pkl'

idx_conv_dict_path=embeds_dir + 'idx_conv_dict.pkl'
################

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
    
    extra_imnet_args = [args.embed_batch_size,
                        args.embed_dim,
                        args.data_dir,
                        args.test_dir,
                        args.folder_to_class_file,
                        args.test_label_file,
                        weights_path,
                        embeds_path,
                        embeds_labels_path,
                        idx_conv_dict_path,
                        test_embeds_path,
                        test_embeds_labels_path]

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

    mnist_df = fed_dmgt_df(mnist_exp(*input_args,
                                     args.mnist_num_sel_rnds),
                           args.trials,
                           args.mnist_num_sel_rnds)
    
    imnet_df = fed_dmgt_df(imnet_exp(*input_args,
                                     *extra_imnet_args,
                                     args.imnet_num_sel_rnds),
                           args.trials,
                           args.imnet_num_sel_rnds)
    
    return mnist_df, imnet_df
