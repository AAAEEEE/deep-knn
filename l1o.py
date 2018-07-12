#!/usr/bin/env python
import pickle
import argparse
import os
import json
import numpy as np
import cupy as cp
import warnings
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from copy import deepcopy

import chainer
import chainer.functions as F

from nlp_utils import convert_seq, convert_snli_seq
from utils import setup_model
from run_dknn import DkNN

def snli_flatten(x):
    '''generate version of x with each token removed'''
    prem, hypo = x
    flatten_hypo = []
    flatten_prem = []
    for i in range(hypo.shape[0]):
        h = np.concatenate((hypo[:i], hypo[i+1:]), axis=0)
        flatten_hypo.append(h)
        flatten_prem.append(prem)

    return (flatten_prem, flatten_hypo)


def flatten(x):
    '''generate version of x with each token removed'''
    assert x.ndim == 1
    assert x.shape[0] > 1
    xs = []
    for i in range(x.shape[0]):
        xs.append(np.concatenate((x[:i], x[i+1:]), axis=0))
    return xs


def leave_one_out(dknn, converter,
                  x,
                  snli=False,
                  use_credibility=True):
    gpu = dknn.model.xp == cp
    device = 0 if gpu else -1
    inputs = converter([x], device=device, with_label=False)
    ys, og_score, _, reg_pred, reg_conf = dknn.predict(inputs, snli=snli)

    xs = snli_flatten(x) if snli else flatten(x)
    batch_size = len(xs[0]) if snli else len(xs)
    y = ys[0] if use_credibility else reg_pred[0]
    ys = [np.array([y], dtype=np.int32) for _ in range(batch_size)]
    inputs = list(zip(xs[0], xs[1], ys)) if snli else list(zip(xs, ys))
    inputs = converter(inputs, device=device)
    xs = inputs['xs']
    ys = inputs['ys']

    # rank
    if use_credibility:
        scores = dknn.get_credibility(xs, ys, use_snli=snli)
        og_score = og_score[0]
    else:
        scores = dknn.get_regular_confidence(xs, ys, snli=snli)
        scores = scores.tolist()
        og_score = reg_conf[0]

    return y, og_score, scores


def vanilla_grad(model, converter,
                 x,
                 snli=False,
                 use_credibility=False):
    gpu = model.xp == cp
    device = 0 if gpu else -1
    inputs = converter([x], device=device, with_label=False)    
    if snli:
        warnings.warn('snli not supported for vanilla grad')
    with chainer.using_config('train', False):
        output = cp.asnumpy(model.predict(inputs, softmax=True))
        y = np.argmax(output)
        original_score = np.max(output)
    onehot_grad = model.get_onehot_grad([x])[0].data.tolist()
    return y, original_score, onehot_grad


def colorize(words, color_array, colors='RdBu'):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1
    cmap = plt.cm.get_cmap(colors)
    template = '<span class="barcode"; style="color: black; \
                background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        if word == '<unk>':
            word = '&ltunk&gt'
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')
    parser.add_argument('--lsh', action='store_true', default=False,
                        help='If true, uses locally sensitive hashing \
                              (with k=10 NN) for NN search.')
    parser.add_argument('--interp_method', type=str, default='dknn',
                        help='choose dknn, softmax, or grad')

    args = parser.parse_args()

    model, train, test, vocab, setup = setup_model(args)
    reverse_vocab = {v: k for k, v in vocab.items()}

    use_snli = False
    if setup['dataset'] == 'snli':
        converter = convert_snli_seq
        colors = 'PiYG'
        use_snli = True
    else:
        converter = convert_seq
        colors = 'RdBu'

    with open(os.path.join(setup['save_path'], 'calib.json')) as f:
        calibration_idx = json.load(f)

    calibration = [train[i] for i in calibration_idx]
    train = [x for i, x in enumerate(train) if i not in calibration_idx]

    '''get dknn layers of training data'''
    dknn = DkNN(model, lsh=args.lsh)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=args.gpu)

    # need to select calibration data more carefully
    '''calibrate the dknn credibility values'''
    dknn.calibrate(calibration, batch_size=setup['batchsize'],
                   converter=converter, device=args.gpu)

    with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
        f.write('<table style="width:100%"> <tr> <th>Method</th> <th>Label</th> <th>Prediction</th> <th>Text</th> </tr>')

    word_importance_scores = defaultdict(lambda: 0)
    word_count = defaultdict(lambda: 0)
    cached_scores = []

    if args.interp_method == 'dknn' or args.interp_method == 'softmax':
        ranker = partial(leave_one_out, dknn, converter)
    elif args.interp_method == 'grad':
        ranker = partial(vanilla_grad, model, converter)
    else:
        exit("Invalid Interpretation Method")

    use_cred = (args.interp_method == 'dknn')

    for i in range(len(test)):
        if use_snli:
            prem, hypo, label = test[i]
            x = (prem, hypo)
        else:
            text, label = test[i]
            x = text
        label = label[0]    

        prediction, original_score, scores = ranker(
                x, snli=use_snli, use_credibility=use_cred)
        sorted_scores = sorted(list(enumerate(scores)), key=lambda x: x[1])
        print('label: {}'.format(label))
        print('prediction: {} ({})'.format(prediction, original_score))

        if use_snli:
            print('Premise: ' + ' '.join(reverse_vocab[w] for w in prem))
            print('Hypothesis: ' + ' '.join(reverse_vocab[w] for w in hypo))
        else:
            print(' '.join(reverse_vocab[w] for w in text))

        for idx, score in sorted_scores:
            if use_snli:
                print(score, reverse_vocab[hypo[idx]])
            else:
                print(score, reverse_vocab[text[idx]])
        print()
        print()
   

        # get drop in score for l10, and flip sign for sentiment
        normalized_scores = []
        words = []
        if not use_snli:
            for idx, score in enumerate(scores):
                if args.interp_method == 'dknn' or args.interp_method == 'softmax':
                    normalized_scores.append(score - original_score)  # for l10 drop in score
                else:
                    normalized_scores.append(score)  # for grad its not a drop
                if snli:
                    words.append(reverse_vocab[hypo[idx]])
                else:
                    words.append(reverse_vocab[text[idx]])
            if not snli and prediction == 1:  # flip sign if positive sentiment
                normalized_scores = [-1 * n for n in normalized_scores]            

        cached_scores.append((words,deepcopy(normalized_scores)))
        # normalize scores across the words, doing positive and negatives seperately        
        total_score_pos = 1e-6    # 1e-6 for case where all positive/neg scores are 0
        total_score_neg = 1e-6
        for idx, s in enumerate(normalized_scores):
            if s < 0:
                total_score_neg = total_score_neg + math.fabs(s)
            else:
                total_score_pos = total_score_pos + s
        for idx, s in enumerate(normalized_scores):
            if s < 0:
                normalized_scores[idx] = (s / total_score_neg) / 2   # / by 2 to get max of -0.5
            else:
                normalized_scores[idx] = (s / total_score_pos) / 2

        # tally individual word influence scores
        for idx, norm_score in enumerate(normalized_scores):
            word_importance_scores[words[idx]] = word_importance_scores[words[idx]] + norm_score
            word_count[words[idx]] = word_count[words[idx]] + 1

        normalized_scores = [0.5 + n for n in normalized_scores]  # center scores
        visual = colorize(words, normalized_scores, colors=colors)  # generate visualize

        # plot snli results        
        if snli:        
            with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
                if label == 0:
                    f.write('Ground Truth Label: Entailment')
                elif label == 1:
                    f.write('Ground Truth Label: Neutral')
                elif label == 2:
                    f.write('Ground Truth Label: Contradiction')                

                if prediction == 0:
                    f.write("Prediction: Entailment ({})         ".format(original_score))
                elif prediction == 1:
                    f.write("Prediction: Neutral ({})         ".format(original_score))
                elif prediction == 2:
                    f.write("Prediction: Contradiction ({})         ".format(original_score))

                f.write("<br>")
                f.write(' '.join(reverse_vocab[w] for w in prem) + '<br>')
                f.write(visual + "<br>")
                f.write("<br>")

        else: # plot sentiment results
            with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
                f.write('<tr>')
                f.write('<td>')
                if args.interp_method == 'dknn':
                    f.write('DkNN Leave-One-Out')
                elif args.interp_method == 'softmax':
                    f.write('Softmax Leave-One-Out')
                else:
                    f.write('Vanilla Gradient')
                f.write('</td>')

                f.write('<td>')
                if label == 1:
                    f.write('Label: Positive')
                else:
                    f.write('Label: Negative')
                f.write('</td>')

                f.write('<td>')
                if prediction == 1:
                    f.write("Prediction: Positive ({0:.2f})         ".format(original_score))
                else:
                    f.write("Prediction: Negative ({0:.2f})         ".format(original_score))
                f.write('</td>')
                
                f.write('<td>')
                f.write(visual)
                f.write('</td>')
                f.write('</tr>')
     
            # print neighbors, not very interpretable for language
            # neighbors = dknn.get_neighbors(x)
            # print('neighbors:')
            # f.write('&nbsp;&nbsp;&nbsp;&nbsp;Nearest Neighbor Sentences: <br>')
            # for neighbor in neighbors[:5]:
            #    curr_nearest_neighbor_input_sentence = '     '
            #    for word in train[neighbor][0]:
            #        curr_nearest_neighbor_input_sentence += reverse_vocab[word] + ' '
            #    print(curr_nearest_neighbor_input_sentence)
            #    f.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + curr_nearest_neighbor_input_sentence + '<br>')
            # f.write('<br>')


    with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f: # end htmp table
        f.write('</table>')

    for word, total_score in list(word_count.items()):
        if total_score < 5:  # 5 or higher to be counted
            del word_count[word]
            del word_importance_scores[word]

    for word, total_score in word_importance_scores.items():
        word_importance_scores[word] = word_importance_scores[word] / word_count[word]

    sorted_by_value = sorted(word_importance_scores.items(), key=lambda kv: kv[1])
    pickle.dump(sorted_by_value, open(args.interp_method + '_sorted.pkl','wb'))
    print(sorted_by_value)

    pickle.dump(cached_scores, open(args.interp_method + '_cached_scores.pkl','wb'))

if __name__ == '__main__':
    main()
