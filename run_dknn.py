#!/usr/bin/env python
import argparse
import json
import sys
from tqdm import tqdm, tqdm_notebook
from collections import Counter

import chainer
import chainer.functions as F

import nets
import text_datasets
from nlp_utils import convert_seq, convert_snli_seq

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree


def setup_model(args):
    sys.stderr.write(json.dumps(args.__dict__, indent=2) + '\n')
    setup = json.load(open(args.model_setup))
    print(type(setup))
    sys.stderr.write(json.dumps(setup, indent=2) + '\n')

    # Load a dataset
    dataset = setup['dataset']
    if dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            char_based=setup['char_based'])
    elif dataset == 'snli':
        train, test, vocab = text_datasets.get_snli(
            char_based=setup['char_based'],
            combine=setup['combine_snli'])
    elif dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=dataset.endswith('.fine'),
            char_based=setup['char_based'])
    elif dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                     'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            dataset, char_based=setup['char_based'])

    # vocab = json.load(open(setup['vocab_path']))
    n_class = setup['n_class']
    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    print('# class: {}'.format(n_class))

    # Setup a model
    if setup['model'] == 'rnn':
        Encoder = nets.RNNEncoder
    elif setup['model'] == 'bilstm':
        Encoder = nets.BiLSTMEncoder
    elif setup['model'] == 'cnn':
        Encoder = nets.CNNEncoder
    elif setup['model'] == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=setup['layer'], n_vocab=len(vocab),
                      n_units=setup['unit'], dropout=setup['dropout'])
    if dataset == 'snli':
        model = nets.SNLIClassifier(encoder, combine=setup['combine_snli'])
    else:
        model = nets.TextClassifier(encoder, n_class)
    chainer.serializers.load_npz(setup['model_path'], model)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    return model, train, test, vocab, setup


class DkNN:

    def __init__(self, model):
        self.model = model
        self.n_dknn_layers = self.model.n_dknn_layers
        self.tree_list = None
        self.label_list = None
        self._A = None

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

        self.tree_list = []  # one lookup tree for each dknn layer
        for i in range(self.n_dknn_layers):
            print('building tree for layer {}'.format(i))
            n_hidden = act_list[i][0].shape[0]
            rbpt = RandomBinaryProjectionTree('rbpt', 75, 75)
            tree = Engine(n_hidden, lshashes=[rbpt])
            for j, example in enumerate(tqdm(act_list[i])):
                assert example.ndim == 1
                assert example.shape[0] == n_hidden
                tree.store_vector(example, j)
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
                knn = self.tree_list[layer_id].neighbours(hidden)
                for nn in knn:
                    neighbors.append(nn[1])

            neighbor_labels = []
            for idx in neighbors:  # for all indices, get their label
                neighbor_labels.append(self.label_list[idx])
            knn_logits .append(neighbor_labels)
        return reg_logits, knn_logits

    def get_credibility(self, xs, ys, calibrated=False):
        assert self.tree_list is not None
        assert self.label_list is not None

        batch_size = len(xs)
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

    def predict(self, xs, calibrated=False):
        assert self.tree_list is not None
        assert self.label_list is not None

        batch_size = len(xs)
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
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')
    args = parser.parse_args()

    model, train, test, vocab, setup = setup_model(args)
    if setup['dataset'] == 'snli' and not setup['combine_snli']:
        converter = convert_snli_seq
    else:
        converter = convert_seq

    '''get dknn layers of training data'''

    dknn = DkNN(model)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=setup['gpu'])

    # need to select calibration data more carefully
    dknn.calibrate(train[:1000], batch_size=setup['batchsize'],
                   converter=converter, device=setup['gpu'])

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

        #  print crdedibility scores and print out the sentence and all its nearest neighbors
        # print(credibility)
        # curr_data_input_sentence = ""
        # for input_words in text[current_position_in_minibatch]:
        #     curr_data_input_sentence += idx2word[input_words] + " "
        # print("Test input", curr_data_input_sentence)

        # print("Nearest Neighbors:")
        # for training_data_index in training_indices:
        #     curr_nearest_neighbor_input = train[training_data_index]
        #     curr_nearest_neighbor_input_sentence = ""
        #     for input_words in curr_nearest_neighbor_input[0]:
        #         curr_nearest_neighbor_input_sentence += idx2word[input_words] + " "
        #     print(curr_nearest_neighbor_input_sentence)

    print('knn accuracy', n_knn_correct / total)
    print('reg accuracy', n_reg_correct / total)

    # TODO
    # calibration for credibility
    # 75 neighbors at each layer, or 75 neighbors total?
    # before or after relu?
    # Consider using a different distance than euclidean, cosine?


if __name__ == '__main__':
    main()
