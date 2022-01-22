import torch 
from torch import nn, optim
from torchvision.models.resnet import ResNet, Bottleneck
from sklearn.isotonic import IsotonicRegression

# modifies resnet50 to accept single-channel images and output 10 classes (for mnist experiment)
class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, bias=False)

# wrapper for resnet (in imagenet experiment, accepts image and outputs penultimate embedding layer instead of softmax scores)
class Embed(nn.Module):
    def __init__(self, model):
        super(Embed, self).__init__()
        self.embed = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        x = self.embed(x)
        return x

# logistic regression model (in imagenet experiment, accepts embedding vector and softmax vector)
class LogRegModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogRegModel, self).__init__()
        self.linear= nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

# deep copies input model
def load_model(model, device, *args):
    
    if len(args) > 0:
        embed_dim, num_classes = args
        model_copy = LogRegModel(embed_dim, num_classes).to(device)
    else:
        model_copy = MnistResNet().to(device)
    
    model_copy.load_state_dict(model.state_dict())
    model_copy = model_copy.to(device)
    return model_copy

# trains model
def train(device, num_epochs, train_loader, model):
   
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        
        train_loss = 0.0
        train_acc = 0.0
        
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (output.max(1)[1]==targets).sum().item()
        
        train_loss = train_loss/len(train_loader.dataset)
        train_acc = train_acc/len(train_loader.dataset)
        
        print(f'Epoch: {epoch} Train Loss: {train_loss:.6f}  Train Acc: {train_acc:.6f}')

        if train_acc >= 0.99:
            break

    return model

# calculates prediction accuracy of input model on all and rare classes of test data
def calc_acc(model, test_loader, num_classes):

    model.eval()
    
    all_acc = []
    rare_acc = []
    
    for test_idx, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)
        
        rare_idxs = (test_y < num_classes/2).nonzero()
        with torch.no_grad():
            
            batch_preds = torch.argmax(model(test_x), dim=1)
            all_acc += [batch_preds.eq(test_y),]
            rare_acc += [batch_preds.eq(test_y)[rare_idxs],]
    
    all_acc = torch.cat(all_acc, dim=0)
    rare_acc = torch.cat(rare_acc, dim=0)

    return rare_acc.float().mean().unsqueeze(0), all_acc.float().mean().unsqueeze(0)

# trains isotonic regressor from input model's top-score prediction on validation data to 0-1 indicator of correc prediction
def train_isoreg(model, val_loader):
    model.eval()

    top_scores = []
    preds_correct = []
    
    for batch_idx, (data, targets) in enumerate(val_loader):
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            logits = model(data)
            softmax = nn.functional.softmax(logits, dim=1)
            
            top_scores += [softmax.max(1).values,]
            
            preds = softmax.max(1).indices
            preds_correct += [(targets==preds).float(),]
    
    top_scores = torch.cat(top_scores, dim=0)
    preds_correct = torch.cat(preds_correct, dim=0)
    
    IsoReg = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip').fit(top_scores.cpu(), preds_correct.cpu())
    return IsoReg 

# value function -- class cardinality
def class_card(prev_labels, x, num_classes, DMGT_model, rare_isoreg, common_isoreg, device):
    
    DMGT_model.eval()
    logits = DMGT_model(x.unsqueeze(0).to(device))
    softmax = nn.functional.softmax(logits, dim=1)
    top_score = softmax.max(1).values
    pred = softmax.max(1).indices
    
    if pred < 5:
        cal_top_score = rare_isoreg.predict(top_score.cpu().detach().numpy())
        
    else:
        cal_top_score = common_isoreg.predict(top_score.cpu().detach().numpy())

    renorm_factor = (1 - torch.from_numpy(cal_top_score).to(device))/(softmax.sum() - top_score) if top_score < 1 else 0
    softmax = torch.mul(renorm_factor*torch.ones(num_classes).to(device), softmax)
    softmax.squeeze().double()[pred] = torch.from_numpy(cal_top_score).to(device)
    softmax = softmax.squeeze()
    label_counts = [(prev_labels==i).float().sum() for i in range(num_classes)]
    
    return sum([softmax[i] * (np.sqrt(label_counts[i] + 1) - np.sqrt(label_counts[i])) for i in range(num_classes)]) 

# selects subsets from stream under DMGT
def get_subsets(stream_x, stream_y, tau, DMGT_model, num_classes, rare_isoreg, common_isoreg, device):
    
    DMGT_x = stream_x[0].unsqueeze(0)
    DMGT_y = stream_y[0].unsqueeze(0)
    
    DMGT_model.eval()
    
    for i in range(1, len(stream_x)):

        if class_card(DMGT_y, stream_x[i], num_classes, DMGT_model, rare_isoreg, common_isoreg, device) > tau:
        
            DMGT_x = torch.cat((DMGT_x, stream_x[i].unsqueeze(0)))
            DMGT_y = torch.cat((DMGT_y, stream_y[i].unsqueeze(0)))
    
    rand_idxs = torch.randperm(len(stream_x))[:len(DMGT_x)]
    RAND_x = stream_x[rand_idxs]
    RAND_y = stream_y[rand_idxs]
    
    return DMGT_x, DMGT_y, RAND_x, RAND_y
