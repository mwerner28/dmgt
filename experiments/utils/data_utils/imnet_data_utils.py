import numpy as np
from numpy import genfromtxt
import torch 
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split, sampler
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models import resnet50
import os
# import Embed model (wrapper around resnet that returns penultimate embedding layer)
from ..exp_utils import Embed

# constructs dataloader for simclr embeddings of imagenet
def get_embed_loader(train_path,
                     class_dict,
                     num_classes,
                     embed_batch_size,
                     num_workers,
                     folder_to_class_file):

    data_transform = Compose([Resize((224, 224)), ToTensor()])
    dataset = ImageFolder(root=train_path, transform=data_transform)
    
    with open(folder_to_class_file) as f:
        lines = list(zip(*[tuple(line.rstrip().split(' '))[:-1] for line in f]))
        keys = lines[0]
        vals = list(map(lambda x: int(x), lines[1]))
    
    temp_dict = dict(zip(keys, vals))
    
    idx_conv_dict = {}
    
    for folder_name in dataset.class_to_idx:
        if dataset.class_to_idx[folder_name] in list(class_dict.keys()):
            idx_conv_dict[dataset.class_to_idx[folder_name]] = temp_dict[folder_name]
    
    rare_classes = list(class_dict.keys())[:int(num_classes/2)]
    common_classes = list(class_dict.keys())[int(num_classes/2):]

    rare_idxs = torch.cat([(torch.tensor(dataset.targets)==cl).nonzero() for cl in rare_classes]).squeeze()
    common_idxs = torch.cat([(torch.tensor(dataset.targets)==cl).nonzero() for cl in common_classes]).squeeze()
    
    embed_idxs = torch.cat((rare_idxs, common_idxs))
    num_pts = len(embed_idxs)
    
    embed_data = Subset(dataset, embed_idxs)
    
    embed_loader = DataLoader(embed_data, batch_size=embed_batch_size, num_workers=num_workers, shuffle=True)

    return embed_loader, num_pts, idx_conv_dict

# constructs imbalanced data stream
def get_datasets(embeds, labels, num_init_pts, imbal, num_classes):
    
    rare_idxs = (labels < 5).nonzero().squeeze()
    common_idxs = (labels >= 5).nonzero().squeeze()

    common_amount = len(common_idxs)
    rare_amount = int(np.floor(common_amount/imbal))
    
    rand_rare_idxs = rare_idxs[torch.randperm(len(rare_idxs))][:rare_amount]
    rand_common_idxs = common_idxs[torch.randperm(len(common_idxs))][:common_amount]
    imbal_idxs = torch.cat((rand_rare_idxs, rand_common_idxs))[torch.randperm(common_amount + rare_amount)]

    embeds, labels = embeds[imbal_idxs], labels[imbal_idxs]
    
    init_dataset, stream_dataset = random_split(TensorDataset(embeds, labels), [num_init_pts, len(imbal_idxs) - num_init_pts])

    return init_dataset, stream_dataset

# gets embeddings of data
def get_embeds(train_path,
               class_dict,
               embed_dim,
               embed_batch_size,
               num_classes,
               num_workers,
               weights_path,
               folder_to_class_file,
               device):
    
    embed_loader, num_pts, idx_conv_dict = get_embed_loader(train_path, class_dict, num_classes, embed_batch_size, num_workers, folder_to_class_file)

    pretrained_model = get_base_model(weights_path, num_classes,device)

    embed_model = Embed(pretrained_model)    
    embed_model.eval()
    embeds = torch.zeros([num_pts, embed_dim])
    embeds_labels = torch.zeros([num_pts])
    
    with torch.no_grad():
        for idx, (data, targets) in enumerate(embed_loader):
            data = data.to(device)
            targets = torch.tensor([class_dict[y.item()] for y in targets]) 
            embeds[idx*embed_batch_size: min(num_pts, (idx+1)*embed_batch_size)] = embed_model(data).squeeze()
            embeds_labels[idx*embed_batch_size: min(num_pts, (idx+1)*embed_batch_size)] = targets
        
    return embeds, embeds_labels, idx_conv_dict

#  constructs dataloaders for test and validation embeddings
def get_test_embed_loaders(embed_dim,
                           embed_batch_size,
                           batch_size,
                           num_classes,
                           num_workers,
                           weights_path,
                           val_path,
                           test_label_file,
                           class_dict,
                           num_test_pts,
                           idx_conv_dict,
                           device):
    
    test_loader = get_test_loader(val_path, test_label_file, idx_conv_dict, batch_size, num_workers, class_dict)
    
    pretrained_model = get_base_model(weights_path, num_classes, device)
    
    embed_model = Embed(pretrained_model)
    
    embed_model.eval()
    
    test_embeds = torch.zeros([num_test_pts, embed_dim])
    test_embeds_labels = torch.zeros([num_test_pts])
    
    inv_dict = {idx_conv_dict[k]:k for k in idx_conv_dict}
    
    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)

            targets = torch.tensor([class_dict[inv_dict[y.item()]] for y in targets]) 
            test_embeds[idx*embed_batch_size: min(num_test_pts, (idx+1)*embed_batch_size)] = embed_model(data).squeeze()
            test_embeds_labels[idx*embed_batch_size: min(num_test_pts, (idx+1)*embed_batch_size)] = targets
        
    rare_embeds_idxs = (test_embeds_labels < num_classes/2).nonzero().squeeze(1)
    common_embeds_idxs = (test_embeds_labels >= num_classes/2).nonzero().squeeze(1)
    
    test_rare_embeds_idxs, val_rare_embeds_idxs = (rare_embeds_idxs[:int(len(rare_embeds_idxs)/2)],
                                                   rare_embeds_idxs[int(len(rare_embeds_idxs)/2):])

    test_common_embeds_idxs, val_common_embeds_idxs = (common_embeds_idxs[:int(len(common_embeds_idxs)/2)],
                                                       common_embeds_idxs[int(len(common_embeds_idxs)/2):])
    
    test_embeds_dataset = TensorDataset(test_embeds, test_embeds_labels)

    test_embeds_loader = DataLoader(Subset(test_embeds_dataset, torch.cat((test_rare_embeds_idxs, test_common_embeds_idxs))),
                                    batch_size=embed_batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)

    rare_val_embeds_loader = DataLoader(Subset(test_embeds_dataset, val_rare_embeds_idxs),
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True)
    
    common_val_embeds_loader = DataLoader(Subset(test_embeds_dataset, val_common_embeds_idxs),
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=True)
    
    val_embeds_loader = DataLoader(Subset(test_embeds_dataset, torch.cat((val_rare_embeds_idxs, val_common_embeds_idxs))),
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=True)
    
    return test_embeds_loader, rare_val_embeds_loader, common_val_embeds_loader, val_embeds_loader

# internal function -- constructs dataloader for imagenet val
def get_test_loader(val_path, test_label_file, idx_conv_dict, batch_size, num_workers, class_dict):

    data_transform = Compose([Resize((224, 224)), ToTensor()])
    
    labels = [int(y) for y in genfromtxt(test_label_file)]
    
    test_dataset = ImageFolder(val_path, transform=data_transform)
    test_dataset.samples = list(map(lambda x, y: (x[0], y), test_dataset.samples, labels))
    test_dataset.targets = labels
    
    class_idxs = torch.cat([(torch.tensor(test_dataset.targets)==cl).nonzero() for cl in list(idx_conv_dict.values())]).squeeze()
    
    test_sampler = sampler.SubsetRandomSampler(class_idxs)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

    return test_loader

# internal function -- constructs resnet50 trained on simclr embeddings 
def get_base_model(weights_path, num_classes, device):
    
    model = resnet50(pretrained=False).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    return model

