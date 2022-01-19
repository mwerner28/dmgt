import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import MNIST

def get_datasets(num_init_pts, imbal, num_classes):

    data_transform = Compose([Resize((224, 224)), ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    mnist_train = MNIST(download=True, train=True, root=STORE MNIST HERE, transform=data_transform)
    
    rare_idxs = (mnist_train.targets < num_classes/2).nonzero().squeeze(1)
    common_idxs = (mnist_train.targets >= num_classes/2).nonzero().squeeze(1)
    
    common_amount = len(common_idxs)
    rare_amount = int(np.floor(common_amount/imbal))
    
    rare_idxs = rare_idxs[torch.randperm(len(rare_idxs))][:rare_amount]
    common_idxs = common_idxs[torch.randperm(len(common_idxs))][:common_amount]
    
    imbal_idxs = torch.cat((rare_idxs, common_idxs))
    imbal_idxs = imbal_idxs[torch.randperm(len(imbal_idxs))]
    
    init_dataset, stream_dataset = random_split(Subset(mnist_train, imbal_idxs), [num_init_pts, len(imbal_idxs) - num_init_pts])
    
    return init_dataset, stream_dataset

def get_val_loaders(num_test_pts, batch_size, num_workers, num_classes):

    transform = Compose([Resize((224, 224)), ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    mnist_test = MNIST(download=True, train=False, root=STORE MNIST HERE, transform=transform)
    
    rare_idxs = (mnist_test.targets < num_classes/2).nonzero().squeeze(1)
    common_idxs = (mnist_test.targets >= num_classes/2).nonzero().squeeze(1)
    
    test_rare_idxs, val_rare_idxs = rare_idxs[:int(len(rare_idxs)/2)], rare_idxs[int(len(rare_idxs)/2):]
    test_common_idxs, val_common_idxs = common_idxs[:int(len(common_idxs)/2)], common_idxs[int(len(common_idxs)/2):]
    
    test_loader = DataLoader(Subset(mnist_test, torch.cat((test_rare_idxs, test_common_idxs))), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    rare_val_loader = DataLoader(Subset(mnist_test, val_rare_idxs), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    common_val_loader = DataLoader(Subset(mnist_test, val_common_idxs), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    val_loader = DataLoader(Subset(mnist_test, torch.cat((val_rare_idxs, val_common_idxs))), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return test_loader, rare_val_loader, common_val_loader, val_loader
