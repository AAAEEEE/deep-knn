import sys
import random
import numpy as np
import torch
from torchtext import data
from args import get_args
from SST1 import SST1Dataset
from utils import clean_str_sst

from sklearn.neighbors import KDTree
from collections import Counter

args = get_args()
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")
np.random.seed(args.seed)
random.seed(args.seed)

if not args.trained_model:
    print("Error: You need to provide a option 'trained_model' to load the model")
    sys.exit(1)

if args.dataset == 'SST-1':
    TEXT = data.Field(batch_first=True, lower=True, tokenize=clean_str_sst)
    LABEL = data.Field(sequential=False)
    train, dev, test = SST1Dataset.splits(TEXT, LABEL)

TEXT.build_vocab(train, min_freq=2)
LABEL.build_vocab(train)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=1, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=1, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config = args
config.target_class = len(LABEL.vocab)
config.words_num = len(TEXT.vocab)
config.embed_num = len(TEXT.vocab)

print("Label dict:", LABEL.vocab.itos)

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage,location: storage)


def predict(dataset_iter, dataset, dataset_name):
    print("Dataset: {}".format(dataset_name))
    model.eval()
    dataset_iter.init_epoch()

    n_correct = 0
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        scores, _ = model(data_batch)
        n_correct += (torch.max(scores, 1)[1].view(data_batch.label.size()).data == data_batch.label.data).sum()

    print("no. correct {} out of {}".format(n_correct, len(dataset)))
    accuracy = 100. * n_correct / len(dataset)
    print("{} accuracy: {:8.6f}%".format(dataset_name, accuracy))


def deep_knn_predict(training_data, dataset_iter, dataset, dataset_name):
    print("Dataset: {}".format(dataset_name))
    model.eval()
    training_data.init_epoch()
    dataset_iter.init_epoch()
    n_correct = 0

    # save activations for all training data
    layer1_act_list = []
    layer2_act_list = []    
    label_list = []
    for data_batch_idx, data_batch in enumerate(training_data):
        layer1_act, layer2_act = model(data_batch)
        pred = torch.max(layer1_act, 1)[1].view(data_batch.label.size()).data.cpu().numpy()        
        layer1_act = layer1_act.data.cpu()
        layer1_act = layer1_act.numpy()
        layer2_act = layer2_act.data.cpu()
        layer2_act = layer2_act.numpy()
        for item in layer1_act:                       
            layer1_act_list.append(item)    
        for item in layer2_act:            
            layer2_act_list.append(item)       
        for item in pred:                    
            label_list.append(item) # TODO, probably doesn't matter, but use predicted output or ground truth label?

    # build KD tree for KNN lookup (TODO use LSH instead)        
    layer1_tree = KDTree(layer1_act_list)
    layer2_tree = KDTree(layer2_act_list)    
    
    # classify examples by nearest examples
    for data_batch_idx, data_batch in enumerate(dataset_iter): 
        if data_batch_idx > 200:
            continue

        layer1_act, layer2_act = model(data_batch)
        pred = torch.max(layer1_act, 1)[1].view(data_batch.label.size()).data.cpu().numpy()
        layer1_act = layer1_act.data.cpu()
        layer1_act = layer1_act.numpy()[0]
        layer2_act = layer2_act.data.cpu()
        layer2_act = layer2_act.numpy()[0]             
        
        _, layer1_indices = layer1_tree.query([layer1_act], k = 75)   # TODO is it 75 neighbors at every layer?, or 75 total
        _, layer2_indices = layer2_tree.query([layer2_act], k = 75)        
       
        # get labels of nearest neighbors
        layer1_labels = []
        layer2_labels = []        

        for item in layer1_indices[0]:            
            layer1_labels.append(label_list[item])
        for item in layer2_indices[0]:
            layer2_labels.append(label_list[item])
        
        # just concat all labels?
        predictions = layer2_labels #layer1_labels + layer2_labels
        most_common,num_most_common = Counter(predictions).most_common(1)[0]
        if most_common == data_batch.label.data.cpu().numpy():
            n_correct = n_correct + 1
            

    print("no. correct {} out of {}".format(n_correct, len(dataset)))
    accuracy = 100. * n_correct / len(dataset)
    print("{} accuracy: {:8.6f}%".format(dataset_name, accuracy))


# # Run the model on the dev set
# predict(dataset_iter=dev_iter, dataset=dev, dataset_name="valid")

# # Run the model on the test set
# predict(dataset_iter=test_iter, dataset=test, dataset_name="test")

deep_knn_predict(training_data = train_iter, dataset_iter=dev_iter, dataset=dev, dataset_name="valid")
