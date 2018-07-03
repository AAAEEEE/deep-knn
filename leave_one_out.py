#!/usr/bin/env python
import argparse
import json
import sys
from tqdm import tqdm, tqdm_notebook
from collections import Counter

import numpy as np
import cupy as cp

import chainer
import chainer.functions as F


import nets
import text_datasets
from nlp_utils import convert_seq, convert_snli_seq
from run_dknn import DkNN, setup_model


def flatten(x):
    '''generate version of x with each token removed'''
    assert x.ndim == 1
    assert x.shape[0] > 1
    if cp.get_array_module(x) == cp:
        x = cp.asnumpy(x)
    xs = []
    for i in range(x.shape[0]):
        xs.append(np.concatenate((x[:i], x[i+1:]), axis=0))
    return xs


def leave_one_out(x, y, scorer, gpu=True):
    # flatten
    assert x.ndim == 1

    if cp.get_array_module(x) == cp:
        x = cp.asnumpy(x)
    xs = flatten(x)
    if gpu:
        xs = [cp.asarray(x) for x in xs]
    ys = [y for _ in xs]

    # rank
    scores = scorer(xs, ys)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')
    args = parser.parse_args()

    model, train, test, vocab, setup = setup_model(args)
    reverse_vocab = {v: k for k, v in vocab.items()}

    if setup['dataset'] == 'snli' and not setup['combine_snli']:
        converter = convert_snli_seq
    else:
        converter = convert_seq

    dknn = DkNN(model)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=setup['gpu'])

    # need to select calibration data more carefully
    dknn.calibrate(train[:1000], batch_size=setup['batchsize'],
                   converter=converter, device=setup['gpu'])

    for i in range(10):
        label = int(test[i][1])
        text = test[i][0]
        y, original_score, _, _, _ = dknn.predict([cp.asarray(text)])
        x = text
        y = y[0]
        scores = leave_one_out(x, y, dknn.get_credibility)
        scores = sorted(list(enumerate(scores)), key=lambda x: x[1])
        print(' '.join(reverse_vocab[w] for w in x))
        print('label: {}'.format(label))
        print('prediction: {} ({})'.format(y, original_score[0]))
        for idx, score in scores[:10]:
            print(score, reverse_vocab[x[idx]])
        print()
        print()


if __name__ == '__main__':
    main()
