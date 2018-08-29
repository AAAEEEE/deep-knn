#!/usr/bin/env python
import argparse
import numpy as np
import cupy as cp
import warnings
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import math

import chainer
import chainer.functions as F

from nlp_utils import convert_seq, convert_snli_seq
from utils import setup_model
from run_dknn import DkNN


# def bigram_flatten(x):
#     '''generate version of x with each bigrams removed'''
#     assert x.ndim == 1
#     assert x.shape[0] > 1
#     if cp.get_array_module(x) == cp:
#         x = cp.asnumpy(x)
#     xs = []
#     for i in range(x.shape[0]):
#         xs.append(np.concatenate((x[:i], x[i+2:]), axis=0))
#     return xs


def snli_flatten(x):
    '''generate version of x with each token removed'''
    # if cp.get_array_module(x) == cp:
    #     x = cp.asnumpy(x)
    reduced_hypo = []
    prem = []
    for i in range(len(x[1][0])):
        reduced_hypo.append(np.concatenate((x[1][0][:i], x[1][0][i+1:]), axis=0))
        prem.append(x[0][0])

    xs = (prem, reduced_hypo)
    return xs


def flatten(x):
    '''generate version of x with each token removed'''
    assert x.ndim == 1
    assert x.shape[0] > 1
    xp = cp.get_array_module(x)
    xs = []
    for i in range(x.shape[0]):
        xs.append(xp.concatenate((x[:i], x[i+1:]), axis=0))
    return xs


def leave_one_out(dknn, x, bigrams=False, snli=False, use_credibility=True):
    if not snli:
        x = [x]
    ys, original_score, _, reg_pred, reg_conf = dknn.predict(x, snli=snli)
    if not snli:
        x = x[0]

    if use_credibility:
        y = ys[0]
    else:
        y = reg_pred[0]

    # gpu = cp.get_array_module(x) == cp
    # x = cp.asnumpy(x)
    if snli:
        xs = snli_flatten(x)
        ys = [int(y) for _ in xs[0]]
    elif bigrams:
        xs = bigram_flatten(x)
        ys = [int(y) for _ in xs]
    else:
        xs = flatten(x)
        ys = [int(y) for _ in xs]
    # if gpu:
    #     xs = [cp.asarray(x) for x in xs]

    # rank
    if use_credibility:
        scores = dknn.get_credibility(xs, ys, use_snli=snli)
        original_score = original_score[0]
        # scores = []
        # for input_x in xs:
        #     scores.append(dknn.get_neighbor_change([input_x], [x]))
    else:
        scores = dknn.get_regular_confidence(xs, ys)
        original_score = reg_conf[0]

    return y, original_score, scores


def vanilla_grad(model, x, bigrams=False, snli=False, use_credibility=False):
    if bigrams:
        warnings.warn('bigrams not supported for vanilla grad')
    if snli:
        warnings.warn('snli not supported for vanilla grad')
    with chainer.using_config('train', False):
        output = model.predict([x], softmax=True)
        y = int(F.argmax(output).data)
        original_score = float(F.max(output).data)
    onehot_grad = model.get_onehot_grad([x])[0].data
    onehot_grad = onehot_grad.tolist()
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
    if setup['dataset'] == 'snli' and not setup['combine_snli']:
        converter = convert_snli_seq
        colors = 'PiYG'
        use_snli = True
    else:
        converter = convert_seq
        colors = 'RdBu'

    dknn = DkNN(model, args.lsh)
    dknn.build(train, batch_size=setup['batchsize'],
               converter=converter, device=args.gpu)

    # need to select calibration data more carefully
    dknn.calibrate(train[:1000], batch_size=setup['batchsize'],
                   converter=converter, device=args.gpu)

    with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
        f.write('<table style="width:100%"> <tr> <th>Method</th> <th>Label</th> <th>Prediction</th> <th>Text</th> </tr>')

    # for j in range(len(test) * 3):
    #     i = j // 3
    #     if args.interp_method == 'dknn':
    #         args.interp_method = 'softmax'
    #     elif args.interp_method == 'softmax':
    #         args.interp_method = 'grad'
    #     elif args.interp_method == 'grad':
    #         args.interp_method = 'dknn'
    if args.interp_method == 'dknn' or args.interp_method == 'softmax':
        ranker = partial(leave_one_out, dknn)
    elif args.interp_method == 'grad':
        ranker = partial(vanilla_grad, model)
    else:
        exit("Method")

    for i in range(len(test)):
        if use_snli:
            prem, hypo, label = test[i]
            x = ([prem], [hypo])
        else:
            text, label = test[i]
            x = cp.asarray(text) #text

        if args.interp_method == 'dknn':
            use_cred = True
        else:
            use_cred = False

        prediction, original_score, scores = ranker(x, snli=use_snli, use_credibility=use_cred)
        sorted_scores = sorted(list(enumerate(scores)), key=lambda x: x[1])
        print('label: {}'.format(label[0]))
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
        # print()
        # # bigrams
        # scores = ranker(x, y, bigrams=True)
        # scores = list(enumerate(scores))
        # sorted_scores = sorted(scores, key=lambda x: x[1])
        # bigrams = [b for l in text for b in zip(x[:-1], x[1:])]
        # for idx, score in sorted_scores[:]:
        #     if score < 1.0:
        #         print(score, reverse_vocab[bigrams[idx][0]] + ' ' + reverse_vocab[bigrams[idx][1]])

        normalized_scores = []
        words = []
        # plot sentiment results visualize results in heatmap
        if not use_snli:
            for idx, score in enumerate(scores):
                if args.interp_method == 'dknn' or args.interp_method == 'softmax':
                    normalized_scores.append(score - original_score)  # for l10 drop in score
                else:
                    normalized_scores.append(score)  # for grad its not a drop
                words.append(reverse_vocab[text[idx]])
            if prediction == 1:  # flip sign if positive
                normalized_scores = [-1 * n for n in normalized_scores]

        # plot snli results visualize results in heatmap
        if use_snli:
            for idx, score in enumerate(scores):
                if args.interp_method == 'dknn' or args.interp_method == 'softmax':
                    normalized_scores.append(score - original_score)  # for l10 drop in score
                else:
                    normalized_scores.append(score)  # for grad its not a drop
                words.append(reverse_vocab[hypo[idx]])
            normalized_scores = [-1 * n for n in normalized_scores] # flip sign so green is drop


        # normalizing for vanilla grad
        # normalize positive and negatives seperately
        # if args.interp_method == 'grad':
        total_score_pos = 1e-6    # 1e-6 for case where all positive/neg scores are 0
        total_score_neg = 1e-6
        for idx, s in enumerate(normalized_scores):
            if s < 0:
                total_score_neg = total_score_neg + math.fabs(s)
            else:
                total_score_pos = total_score_pos + s
        for idx, s in enumerate(normalized_scores):
            if s < 0:
                normalized_scores[idx] = (s / total_score_neg) / 2
            else:
                normalized_scores[idx] = (s / total_score_pos) / 2

        # normalize total score for vanilla grad
        # if args.interp_method == 'grad':
        #     total_score = 0
        #     for idx, s in enumerate(normalized_scores):
        #         total_score = total_score + math.fabs(s)
        #     for idx, s in enumerate(normalized_scores):
        #         normalized_scores[idx] = (s / total_score) / 2

        normalized_scores = [0.5 + n for n in normalized_scores]  # center scores


        # visual = colorize(words, normalized_scores, colors=colors)
        # with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
        #     f.write('<tr>')
        #     f.write('<td>')
        #     if args.interp_method == 'dknn':
        #         f.write('DkNN Leave-One-Out')
        #     elif args.interp_method == 'softmax':
        #         f.write('Softmax Leave-One-Out')
        #     else:
        #         f.write('Vanilla Gradient')
        #     f.write('</td>')

        #     f.write('<td>')
        #     if label == 1:
        #         f.write('Label: Positive')
        #     else:
        #         f.write('Label: Negative')
        #     f.write('</td>')

        #     f.write('<td>')
        #     if prediction == 1:
        #         f.write("Prediction: Positive ({0:.2f})         ".format(original_score))
        #     else:
        #         f.write("Prediction: Negative ({0:.2f})         ".format(original_score))
        #     f.write('</td>')

        #     f.write('<td>')
        #     f.write(visual)
        #     f.write('</td>')

        #     f.write('</tr>')

        # plot snli results
        visual = colorize(words, normalized_scores, colors = colors)
        with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
            if label == 0:
                f.write('Ground Truth Label: Entailment')
            elif label == 1:
                f.write('Ground Truth Label: Neutral')
            elif label == 2:
                f.write('Ground Truth Label: Contradiction')
            else:
                exit("Label not found")

            if prediction == 0:
                f.write("Prediction: Entailment ({})         ".format(original_score))
            elif prediction == 1:
                f.write("Prediction: Neutral ({})         ".format(original_score))
            elif prediction == 2:
                f.write("Prediction: Contradiction ({})         ".format(original_score))
            else:
                eixt("Prediction Label not found")
            f.write("<br>")
            f.write(' '.join(reverse_vocab[w] for w in prem) + '<br>')
            f.write(visual + "<br>")

        # display scores normalized across words
        # normalized_scores = []
        # for idx, score in scores[:]:
        #     normalized_scores.append(score - original_score)
        # if prediction == 1:  # flip sign if positive
        #     normalized_scores = [-1 * n for n in normalized_scores]
        # total_score = 1e-6
        # for p in normalized_scores:
        #     total_score += math.fabs(p)
        # normalized_scores = [0.5 + n / total_score for n in normalized_scores]
        # visual = colorize(words, normalized_scores)
        # with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
        #     if label == 1:
        #         f.write('Ground Truth Label: Positive&nbsp;')
        #     else:
        #         f.write('Ground Truth Label: Negative&nbsp;')
        #     if prediction == 1:
        #         f.write("Prediction: Positive ({})         ".format(original_score))
        #     else:
        #         f.write("Prediction: Negative ({})         ".format(original_score))
        #     f.write(visual + "<br>")

            # print neighbors
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

            print()
            print()


    with open(setup['dataset'] + '_' + setup['model'] + '_colorize.html', 'a') as f:
        f.write('</table>')

if __name__ == '__main__':
    main()
