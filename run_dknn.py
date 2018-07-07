#!/usr/bin/env python
import argparse
from tqdm import tqdm, tqdm_notebook
from collections import Counter

import chainer
import chainer.functions as F

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree
from sklearn.neighbors import KDTree

from nlp_utils import convert_seq, convert_snli_seq
from utils import setup_model


class DkNN:

    def __init__(self, model, lsh=False):
        self.model = model
        self.n_dknn_layers = self.model.n_dknn_layers
        self.tree_list = None
        self.label_list = None
        self._A = None
        self.lsh = lsh

    def build(self, train, batch_size=64, converter=convert_seq, device=0):
        train_iter = chainer.iterators.SerialIterator(
                train, batch_size, repeat=False)
        train_iter.reset()

        act_list = [[] for _ in range(self.n_dknn_layers)]
        label_list = []
        print('caching hiddens')
        n_batches = len(train) // batch_size
        for i, train_batch in enumerate(tqdm(train_iter, total=n_batches)):
            data = converter(train_batch, device=device, with_label=True)
            text = data['xs']
            labels = data['ys']

            with chainer.using_config('train', False):
                _, dknn_layers = self.model.predict(text, dknn=True)
                assert len(dknn_layers) == self.model.n_dknn_layers
            for i in range(self.n_dknn_layers):
                layer = dknn_layers[i]
                layer.to_cpu()
                act_list[i] += [x for x in layer.data]
            label_list.extend([int(x) for x in labels])
        self.act_list = act_list
        self.label_list = label_list

        if self.lsh:
            print('using Locally Sensitive Hashing for NN Search')
        else:
            print('using KDTree for NN Search')
        self.tree_list = []  # one lookup tree for each dknn layer
        for i in range(self.n_dknn_layers):
            print('building tree for layer {}'.format(i))
            if self.lsh:  # if lsh
                n_hidden = act_list[i][0].shape[0]
                rbpt = RandomBinaryProjectionTree('rbpt', 75, 75)
                tree = Engine(n_hidden, lshashes=[rbpt])

                for j, example in enumerate(tqdm(act_list[i])):
                    assert example.ndim == 1
                    assert example.shape[0] == n_hidden

                    tree.store_vector(example, j)
            else:  # if kdtree
                tree = KDTree(act_list[i])

            self.tree_list.append(tree)

    def calibrate(self, data, batch_size=64, converter=convert_seq, device=0):
        data_iter = chainer.iterators.SerialIterator(
                data, batch_size, repeat=False)
        data_iter.reset()

        print('calibrating credibility')
        self._A = []
        n_batches = len(data) // batch_size
        for i, batch in enumerate(tqdm(data_iter, total=n_batches)):
            batch = converter(batch, device=device, with_label=True)
            labels = [int(x) for x in batch['ys']]
            _, knn_logits = self(batch['xs'])
            for j, _ in enumerate(batch):
                cnt_all = len(knn_logits[j])
                preds = dict(Counter(knn_logits[j]).most_common())
                cnt_y = preds[labels[j]]
                self._A.append(cnt_y / cnt_all)

    def get_neighbor_change(self, xs, x):
        full_length_neighbors = self.get_neighbors(x)        
        l10_neighbors = self.get_neighbors(xs)        
        overlap = 0.0
        for i in l10_neighbors:
            if i in full_length_neighbors:
                overlap = overlap + 1
        return overlap / len(l10_neighbors)

    def get_neighbors(self, xs):
        assert self.tree_list is not None
        assert self.label_list is not None

        with chainer.using_config('train', False):
            reg_logits, dknn_layers = self.model.predict(
                    xs, softmax=True, dknn=True)

        _dknn_layers = []
        for layer in dknn_layers:
            layer.to_cpu()
            _dknn_layers.append([x for x in layer.data])
        # n_examples * n_layers
        dknn_layers = list(map(list, zip(*_dknn_layers)))

        for i, example_layers in enumerate(dknn_layers):
            # go through examples in the batch
            neighbors = []
            for layer_id, hidden in enumerate(example_layers):
                # go through layers and get neighbors for each
                if self.lsh:  # use lsh
                    knn = self.tree_list[layer_id].neighbours(hidden)
                    for nn in knn:
                        neighbors.append(nn[1])
                else:  # use kdtree
                    _, knn = self.tree_list[layer_id].query([hidden], k=75)
                    neighbors = knn[0]
        return neighbors

    def __call__(self, xs):
        assert self.tree_list is not None
        assert self.label_list is not None

        with chainer.using_config('train', False):
            reg_logits, dknn_layers = self.model.predict(
                    xs, softmax=True, dknn=True)

        _dknn_layers = []
        for layer in dknn_layers:
            layer.to_cpu()
            _dknn_layers.append([x for x in layer.data])
        # n_examples * n_layers
        dknn_layers = list(map(list, zip(*_dknn_layers)))

        knn_logits = []
        for i, example_layers in enumerate(dknn_layers):
            # go through examples in the batch
            neighbors = []
            for layer_id, hidden in enumerate(example_layers):
                # go through layers and get neighbors for each
                if self.lsh:  # use lsh
                    knn = self.tree_list[layer_id].neighbours(hidden)
                    for nn in knn:
                        neighbors.append(nn[1])
                else:  # use kdtree
                    _, knn = self.tree_list[layer_id].query([hidden], k=75)
                    neighbors = knn[0]

            neighbor_labels = []
            for idx in neighbors:  # for all indices, get their label
                neighbor_labels.append(self.label_list[idx])
            knn_logits .append(neighbor_labels)
        return reg_logits, knn_logits

    def get_credibility(self, xs, ys, calibrated=False, use_snli=False):
        assert self.tree_list is not None
        assert self.label_list is not None

        batch_size = len(xs)
        if use_snli:
            batch_size = len(xs[0])        

        _, knn_logits = self(xs)

        ys = [int(y) for y in ys]
        knn_cred = []

        for i in range(batch_size):
            cnt_all = len(knn_logits[i])
            cnts = dict(Counter(knn_logits[i]).most_common())
            p_1 = cnts.get(ys[i], 0) / cnt_all
            if calibrated and self._A is not None:
                p_1 = len([x for x in self._A if x >= p_1]) / len(self._A)
            knn_cred.append(p_1)
        return knn_cred

    def get_regular_confidence(self, xs, snli=False):
        reg_logits, knn_logits = self(xs)
        # reg_pred = F.argmax(reg_logits, 1).data.tolist()
        reg_conf = F.max(reg_logits, 1).data.tolist()
        return reg_conf

    def predict(self, xs, calibrated=False, snli=False):
        assert self.tree_list is not None
        assert self.label_list is not None

        batch_size = len(xs)                
        if snli:
            batch_size = len(xs[0])
        
        reg_logits, knn_logits = self(xs)

        reg_pred = F.argmax(reg_logits, 1).data.tolist()
        reg_conf = F.max(reg_logits, 1).data.tolist()

        knn_pred, knn_cred, knn_conf = [], [], []
        for i in range(batch_size):
            cnt_all = len(knn_logits[i])
            cnts = Counter(knn_logits[i]).most_common()
            label, cnt_1st = cnts[0]
            if len(cnts) > 1:
                _, cnt_2nd = cnts[1]
            else:
                cnt_2nd = 0
            p_1 = cnt_1st / cnt_all
            p_2 = cnt_2nd / cnt_all
            if calibrated and self._A is not None:
                p_1 = len([x for x in self._A if x >= p_1]) / len(self._A)
                p_2 = len([x for x in self._A if x >= p_2]) / len(self._A)
            knn_pred.append(label)
            knn_cred.append(p_1)
            knn_conf.append(1 - p_2)
        return knn_pred, knn_cred, knn_conf, reg_pred, reg_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')
    parser.add_argument('--lsh', action='store_true', default=False,
                        help='If true, uses locally sensitive hashing \
                              (with k=10 NN) for NN search.')
    args = parser.parse_args()

    model, train, test, vocab, setup = setup_model(args)
    if setup['dataset'] == 'snli' and not setup['combine_snli']:
        converter = convert_snli_seq
    else:
        converter = convert_seq

    '''get dknn layers of training data'''
    dknn = DkNN(model, lsh=args.lsh)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=args.gpu)

    # need to select calibration data more carefully
    dknn.calibrate(train[:1000], batch_size=setup['batchsize'],
                   converter=converter, device=args.gpu)

    # activation_tree = KDTree(act_list)

    '''run dknn on evaluation data'''
    test_iter = chainer.iterators.SerialIterator(
            test, setup['batchsize'], repeat=False)
    test_iter.reset()

    print('run dknn on evaluation data')

    total = 0
    n_reg_correct = 0
    n_knn_correct = 0
    n_batches = len(test) // setup['batchsize']
    for test_batch in tqdm(test_iter, total=n_batches):
        data = converter(test_batch, device=args.gpu, with_label=True)
        text = data['xs']
        knn_pred, knn_cred, knn_conf, reg_pred, reg_conf = dknn.predict(text)
        label = [int(x) for x in data['ys']]
        total += len(label)
        n_knn_correct += sum(x == y for x, y in zip(knn_pred, label))
        n_reg_correct += sum(x == y for x, y in zip(reg_pred, label))

    print('knn accuracy', n_knn_correct / total)
    print('reg accuracy', n_reg_correct / total)

    # TODO
    # 75 neighbors at each layer, or 75 neighbors total?
    # before or after relu?
    # Consider using a different distance than euclidean, cosine?


if __name__ == '__main__':
    main()
