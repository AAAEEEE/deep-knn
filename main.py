from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


from sklearn.neighbors import KDTree
from collections import Counter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # TODO, are the "representations" in the paper before or after non-linearities
    # TODO, does he also consider the final layer? isn't that just the logits which will obviously break for adversarial examples?
    def forward(self, x):
        layer1_act = F.relu(F.max_pool2d(self.conv1(x), 2))
        layer2_act = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(layer1_act)), 2))
        layer3_act = layer2_act.view(-1, 320)
        layer3_act = F.relu(self.fc1(layer3_act))
        layer3_act_dropped = F.dropout(layer3_act, training=self.training)
        layer4_act = self.fc2(layer3_act_dropped)
        return F.log_softmax(layer4_act, dim=1), layer1_act, layer2_act, layer3_act, layer4_act

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, _, _, _, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0

    # save activations for all training data
    layer1_act_list = []
    layer2_act_list = []
    layer3_act_list = []
    layer4_act_list = []
    label_list = []
    for batch_idx, (data, target) in enumerate(train_loader):        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)      

        output, layer1_act, layer2_act, layer3_act, layer4_act = model(data)
        output = output.data.cpu()
        layer1_act = layer1_act.data.cpu()
        layer1_act = layer1_act.view(-1, 1440).numpy() # TODO how does he reshape?
        layer2_act = layer2_act.data.cpu()
        layer2_act = layer2_act.view(-1, 320).numpy()
        layer3_act = layer3_act.data.cpu().numpy()
        layer4_act = layer4_act.data.cpu().numpy()
        pred = output.max(1, keepdim=True)[1].numpy()        
        for item in layer1_act:                                    
            layer1_act_list.append(item)    
        for item in layer2_act:            
            layer2_act_list.append(item)    
        for item in layer3_act:            
            layer3_act_list.append(item)    
        for item in layer4_act:            
            layer4_act_list.append(item)    
        for item in pred:                        
            label_list.append(item[0]) # TODO, probably doesn't matter, but use predicted output or ground truth label?

    # build KD tree for KNN lookup (TODO use LSH instead)    
    layer1_tree = KDTree(layer1_act_list)
    layer2_tree = KDTree(layer2_act_list)
    layer3_tree = KDTree(layer3_act_list)
    layer4_tree = KDTree(layer4_act_list)    
    
    # classify examples by nearest examples
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # run data through model and get activations, then find indices of nearest neighbors
        data, target = Variable(data, volatile=True), Variable(target)
        output, layer1_act, layer2_act, layer3_act, layer4_act = model(data)
        output = output.data.cpu()
        layer1_act = layer1_act.data.cpu()
        layer1_act = layer1_act.view(-1, 1440).numpy()[0] # TODO how does he reshape?
        layer2_act = layer2_act.data.cpu()
        layer2_act = layer2_act.view(-1, 320).numpy()[0]
        layer3_act = layer3_act.data.cpu().numpy()[0]
        layer4_act = layer4_act.data.cpu().numpy()[0]
        
        _, layer1_indices = layer1_tree.query([layer1_act], k = 75)   # TODO is it 75 neighbors at every layer?, or 75 total
        _, layer2_indices = layer2_tree.query([layer2_act], k = 75)
        _, layer3_indices = layer3_tree.query([layer3_act], k = 75)
        _, layer4_indices = layer4_tree.query([layer4_act], k = 75)
       
        # get labels of nearest neighbors
        layer1_labels = []
        layer2_labels = []
        layer3_labels = []
        layer4_labels = []

        for item in layer1_indices[0]:            
            layer1_labels.append(label_list[item])
        for item in layer2_indices[0]:
            layer2_labels.append(label_list[item])
        for item in layer3_indices[0]:
            layer3_labels.append(label_list[item])
        for item in layer4_indices[0]:
            layer4_labels.append(label_list[item])

        # just concat all labels?
        predictions = layer1_labels + layer2_labels + layer3_labels + layer4_labels
        most_common,num_most_common = Counter(predictions).most_common(1)[0] # 4, 6 times    
        if most_common == target.data.cpu().numpy():
            correct = correct + 1
    
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)

test()