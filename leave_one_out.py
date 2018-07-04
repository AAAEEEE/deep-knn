#!/usr/bin/env python
import argparse
import json
import sys
from tqdm import tqdm, tqdm_notebook
from collections import Counter

import numpy as np
#import cupy as cp

import matplotlib
import matplotlib.pyplot as plt


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
    # if cp.get_array_module(x) == cp:
    #     x = cp.asnumpy(x)
    xs = []
    for i in range(x.shape[0]):
        xs.append(np.concatenate((x[:i], x[i+1:]), axis=0))
    return xs


def leave_one_out(x, y, scorer, gpu=True):
    # flatten
    assert x.ndim == 1

    # if cp.get_array_module(x) == cp:
    #     x = cp.asnumpy(x)
    xs = flatten(x)
    # if gpu:
    #     xs = [cp.asarray(x) for x in xs]
    ys = [y for _ in xs]

    # rank
    scores = scorer(xs, ys)
    return scores

def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')
    parser.add_argument('--lsh', action='store_true', default = False,
                        help='If true, uses locally sensitive hashing (with k = 10 NN) for NN search.')
    args = parser.parse_args()
    
    model, train, test, vocab, setup = setup_model(args)
    reverse_vocab = {v: k for k, v in vocab.items()}

    if setup['dataset'] == 'snli' and not setup['combine_snli']:
        converter = convert_snli_seq
    else:
        converter = convert_seq

    dknn = DkNN(model, args.lsh)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=setup['gpu'])

    # need to select calibration data more carefully    
    dknn.calibrate(train[:1000], batch_size=setup['batchsize'],
                   converter=converter, device=setup['gpu'])

    for i in range(20): 
        label = int(test[i][1])
        text = test[i][0]
        y, original_score, _, _, _ = dknn.predict([text])
        x = text
        y = y[0]
        scores = leave_one_out(x, y, dknn.get_credibility)
        scores = list(enumerate(scores))
        sorted_scores = sorted(scores, key=lambda x: x[1])                          
        
        # visualize results in heatmap
        normalized_scores = []
        words = []
        for idx, score in scores[:]:
            normalized_scores.append(score - original_score[0])            
            words.append(reverse_vocab[x[idx]])

        total_score = np.sum(normalized_scores)
        normalized_scores = [0.5 + n / total_score for n in normalized_scores]

        visual = colorize(words, normalized_scores)
        with open('colorize.html', 'a') as f:
            if y == 1:
                f.write("Positive ({})         ".format(original_score[0]))

                print(' '.join(reverse_vocab[w] for w in x))
                print('label: {}'.format(label))
                print('prediction: {} ({})'.format(y, original_score[0]))          
                for idx, score in sorted_scores[:]:
                    if score < 1.0:
                        print(score, reverse_vocab[x[idx]])

            else:
                continue
                f.write("Negative ({})         ".format(original_score[0]))

            f.write(visual + "<br>")

        # print neighbors
        neighbors = dknn.get_neighbors([text])
        print('neighbors:')
        for neighbor in neighbors[:5]:
            curr_nearest_neighbor_input_sentence = '     '
            for word in train[neighbor][0]:            
                curr_nearest_neighbor_input_sentence += reverse_vocab[word] + ' '
            print(curr_nearest_neighbor_input_sentence)
        print()
        print()


if __name__ == '__main__':
    main()
