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
import chainer.functions as f

from nlp_utils import convert_seq, convert_snli_seq
from utils import setup_model
from run_dknn import dknn

'''generate a batch of snli hypothesis, x, each entry with a different word left out'''
def snli_flatten(x):
    prem, hypo = x
    flatten_hypo = []
    flatten_prem = []
    for i in range(hypo.shape[0]):
        h = np.concatenate((hypo[:i], hypo[i+1:]), axis=0)
        flatten_hypo.append(h)
        flatten_prem.append(prem)
    return (flatten_prem, flatten_hypo)

'''generate a batch of examples, x, each entry with a different word left out'''
def flatten(x):
    assert x.ndim == 1
    assert x.shape[0] > 1
    xs = []
    for i in range(x.shape[0]):
        xs.append(np.concatenate((x[:i], x[i+1:]), axis=0))
    return xs

''' performs leave one out interpretations. has multiple options for snli (paired inputs)
or single input tasks. also has options for using dknn credibility or confidence'''
def leave_one_out(dknn, converter,
                  x,
                  snli=false,
                  use_credibility=true):
    gpu = dknn.model.xp == cp
    device = 0 if gpu else -1
    inputs = converter([x], device=device, with_label=false)  # setup gpu stuff
    ys, og_score, _, reg_pred, reg_conf = dknn.predict(inputs, snli=snli)  # get original prediction

    xs = snli_flatten(x) if snli else flatten(x)    # batch of leave out one word
    batch_size = len(xs[0]) if snli else len(xs)
    y = ys[0] if use_credibility else reg_pred[0] # get prediction depending on mode
    ys = [np.array([y], dtype=np.int32) for _ in range(batch_size)]
    inputs = list(zip(xs[0], xs[1], ys)) if snli else list(zip(xs, ys))
    inputs = converter(inputs, device=device)
    xs = inputs['xs']
    ys = inputs['ys']
    
    if use_credibility:  # if dknn, then get scores of each input with words left out
        scores = dknn.get_credibility(xs, ys, use_snli=snli)
        og_score = og_score[0]
    else:
        scores = dknn.get_regular_confidence(xs, ys, snli=snli)
        scores = scores.tolist()
        og_score = reg_conf[0]

    return y, og_score, scores

''' does gradient based interpretations'''
def vanilla_grad(model, converter,
                 x,
                 snli=false,
                 use_credibility=false):
    gpu = model.xp == cp
    device = 0 if gpu else -1
    inputs = converter([x], device=device, with_label=false)    
    if snli:
        warnings.warn('snli not supported for vanilla grad')
    with chainer.using_config('train', false):
        output = cp.asnumpy(model.predict(inputs, softmax=true))
        y = np.argmax(output)
        original_score = np.max(output)
    onehot_grad = model.get_onehot_grad([x])[0].data.tolist()
    return y, original_score, onehot_grad

''' generates saliency map visualizations as seen in the paper'''
def colorize(words, color_array, colors='rdbu'):
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
    parser = argparse.argumentparser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='gpu id (negative value indicates cpu)')
    parser.add_argument('--model-setup', required=true,
                        help='model setup dictionary.')
    parser.add_argument('--lsh', action='store_true', default=false,
                        help='if true, uses locally sensitive hashing \
                              (with k=10 nn) for nn search.')
    parser.add_argument('--interp_method', type=str, default='dknn',
                        help='choose dknn, softmax, or grad')

    args = parser.parse_args()

    model, train, test, vocab, setup = setup_model(args)
    reverse_vocab = {v: k for k, v in vocab.items()}

    use_snli = false
    if setup['dataset'] == 'snli':  # if snli, change colors and set flags
        converter = convert_snli_seq
        colors = 'piyg'  
        use_snli = true
    else:
        converter = convert_seq
        colors = 'rdbu'

    with open(os.path.join(setup['save_path'], 'calib.json')) as f:
        calibration_idx = json.load(f)

    calibration = [train[i] for i in calibration_idx]
    train = [x for i, x in enumerate(train) if i not in calibration_idx]

    '''get dknn layers of training data'''
    dknn = dknn(model, lsh=args.lsh)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=args.gpu)

    '''calibrate the dknn credibility values'''
    dknn.calibrate(calibration, batch_size=setup['batchsize'],
                   converter=converter, device=args.gpu)

    # opens up a html file for printing results. writes a table header to make it pretty
    with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
        f.write('<table style="width:100%"> <tr> <th>method</th> <th>label</th> <th>prediction</th> <th>text</th> </tr>')

    # setup word importance ranking function depending on mode
    if args.interp_method == 'dknn' or args.interp_method == 'softmax':
        ranker = partial(leave_one_out, dknn, converter)
    elif args.interp_method == 'grad':
        ranker = partial(vanilla_grad, model, converter)

    use_cred = (args.interp_method == 'dknn')

    for i in range(len(test)):  # generate interpretations for the whole test set
        if use_snli:
            prem, hypo, label = test[i]
            x = (prem, hypo)
        else:
            text, label = test[i]
            x = text
        label = label[0]    

        prediction, original_score, scores = ranker(
                x, snli=use_snli, use_credibility=use_cred)  # get original score, and scores for all individual words 
        sorted_scores = sorted(list(enumerate(scores)), key=lambda x: x[1]) # sort scores for each word
        print('label: {}'.format(label))
        print('prediction: {} ({})'.format(prediction, original_score))

        if use_snli:  # print out inputs
            print('premise: ' + ' '.join(reverse_vocab[w] for w in prem))
            print('hypothesis: ' + ' '.join(reverse_vocab[w] for w in hypo))
        else:
            print(' '.join(reverse_vocab[w] for w in text))

        for idx, score in sorted_scores: # print word importances
            if use_snli:
                print(score, reverse_vocab[hypo[idx]])
            else:
                print(score, reverse_vocab[text[idx]])
        print()
        print()
   

        # if using l10, get drop in score.
        normalized_scores = []
        words = []        
        for idx, score in enumerate(scores):
            if args.interp_method == 'dknn' or args.interp_method == 'softmax':
                normalized_scores.append(score - original_score)  # for l10 drop in score
            else:
                normalized_scores.append(score)  # for grad its not a drop
            if snli:
                words.append(reverse_vocab[hypo[idx]])
            else:
                words.append(reverse_vocab[text[idx]])
        # flip sign if positive sentiment. i.e., for positive class, drop in score = red highlight.
        # for negative class, drop is score = blue highlight
        if not snli and prediction == 1:  
            normalized_scores = [-1 * n for n in normalized_scores]            
        if snli:
            normalized_scores = [-1 * n for n in normalized_scores] # flip sign so green is drop
        
        # normalize scores across the words, doing positive and negatives seperately        
        # final scores should be in range [0,1] 0 is dark red, 1 is dark blue. 0.5 is no highlight
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
        normalized_scores = [0.5 + n for n in normalized_scores]  # center scores
        
        visual = colorize(words, normalized_scores, colors=colors)  # generate saliency map colors

        # setup html table row with snli results        
        if snli:        
            with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
                if label == 0:
                    f.write('ground truth label: entailment')
                elif label == 1:
                    f.write('ground truth label: neutral')
                elif label == 2:
                    f.write('ground truth label: contradiction')                

                if prediction == 0:
                    f.write("prediction: entailment ({})         ".format(original_score))
                elif prediction == 1:
                    f.write("prediction: neutral ({})         ".format(original_score))
                elif prediction == 2:
                    f.write("prediction: contradiction ({})         ".format(original_score))

                f.write("<br>")
                f.write(' '.join(reverse_vocab[w] for w in prem) + '<br>')
                f.write(visual + "<br>")
                f.write("<br>")

        # setup html table row with sentiment results        
        else: 
            with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
                f.write('<tr>')
                f.write('<td>')
                if args.interp_method == 'dknn':
                    f.write('conformity leave-one-out')
                elif args.interp_method == 'softmax':
                    f.write('confidence leave-one-out')
                else:
                    f.write('vanilla gradient')
                f.write('</td>')

                f.write('<td>')
                if label == 1:
                    f.write('label: positive')
                else:
                    f.write('label: negative')
                f.write('</td>')

                f.write('<td>')
                if prediction == 1:
                    f.write("prediction: positive ({0:.2f})         ".format(original_score))
                else:
                    f.write("prediction: negative ({0:.2f})         ".format(original_score))
                f.write('</td>')
                
                f.write('<td>')
                f.write(visual)
                f.write('</td>')
                f.write('</tr>')
     
        # print nearest neighbor training data points for interpretation by analogy
        # neighbors = dknn.get_neighbors(x)
        # print('neighbors:')        
        # for neighbor in neighbors[:5]:
        #    curr_nearest_neighbor_input_sentence = '     '
        #    for word in train[neighbor][0]:
        #        curr_nearest_neighbor_input_sentence += reverse_vocab[word] + ' '
        #    print(curr_nearest_neighbor_input_sentence)        

    with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f: # end html table
        f.write('</table>')

if __name__ == '__main__':
    main()
