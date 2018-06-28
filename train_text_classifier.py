#!/usr/bin/env python
import argparse
import datetime
import json
import os
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import chainer.functions as F

import nets
from nlp_utils import convert_seq
import text_datasets
from sklearn.neighbors import KDTree
from collections import Counter


def create_parser():
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--dataset', '-data', default='stsa.binary',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type.')
    parser.add_argument('--char-based', action='store_true')
    parser.add_argument('--word_vectors', default=None,
                        help='word vector directory')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
    current_datetime = '{}'.format(datetime.datetime.today())

    # Load a dataset
    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            char_based=args.char_based)
    elif args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=args.dataset.endswith('.fine'),
            char_based=args.char_based)
    elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            args.dataset, char_based=args.char_based)

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Setup a model
    if args.model == 'rnn':
        Encoder = nets.RNNEncoder
    elif args.model == 'cnn':
        Encoder = nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)
    model = nets.TextClassifier(encoder, n_class)

    # load word vectors
    if args.word_vectors:
        print("loading word vectors")
        with open(args.word_vectors, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = model.xp.array(line_list[1::], dtype=np.float32)
                    model.encoder.embed.W.data[vocab[word]] = vec
    else:
        print("WARNING: NO Word Vectors")

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        test_iter, model,
        converter=convert_seq, device=args.gpu))

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    current = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(current, args.out, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(current, args.out, 'best_model.npz')
    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

    idx2word = {}   # build reverse dict
    for word, idx in vocab.items():
        idx2word[idx] = word

    # Run the training
    trainer.run()

    # run deep knn on training data and store activations
    act_list = []  # all the activations, layer[training data [allpoints] a list of lists of activations
    label_list = []

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, repeat = False) # no repeat to make it easy to save all datapoints
    train_iter.reset()

    for train_batch in train_iter:
        data = convert_seq(train_batch, device=args.gpu, with_label = True)
        text = data['xs']
        labels = data['ys']
        # run forward pass of data

        with chainer.using_config('train', False), chainer.no_backprop_mode():   # TODO, is dropout off now?
            output, activations = model.deep_knn_predict(text)
            output.to_cpu()
            activations.to_cpu()

            #  add predicted label to list
            for prediction in output:
                label_list.append(model.xp.argmax(prediction.data)) # should this be predicted label or ground truth?

            # activations is (num_layers, batch_size, embed_size), make it be (batch_size, num_layers, embed_size)
            activations = F.expand_dims(activations,axis = 1)
            #activations = activations.reshape(activations.shape[1], activations.shape[0], activations.shape[2])
            for activation in activations:
                act_list.append(activation.data)  # each entry in act_list is (num_layers, embed_size)


    from nearpy import Engine
    from nearpy.hashes import RandomBinaryProjectionTree

    tree_list = []    # there is one lookup knn tree for each layer of the network

    num_layers = args.layer
    if args.model == 'cnn':  # they don't count the cnn as a layer, only the mlps
        num_layers = num_layers + 1

    num_layers = 1
    print("WARNING NUM LAYERS NEEDS TO BE REMOVED SON!!!!")
    for layer in range(num_layers):
        num_dimensions = act_list[0][layer].shape[0]  # for all the layers, get the embed_size of that layer
        rbpt = RandomBinaryProjectionTree('rbpt', 75, 75)
        activation_tree = Engine(num_dimensions, lshashes=[rbpt])
        tree_list.append(activation_tree)

    for ind, data_point in enumerate(act_list):
        for layer in range(data_point.shape[0]):
            tree_list[layer].store_vector(data_point[layer], ind)

    #activation_tree = KDTree(act_list)

    # run deep knn on evaluation data
    total = 0
    n_correct = 0
    test_iter.reset()
    for test_batch in test_iter:
        data = convert_seq(test_batch, device=args.gpu, with_label = True)
        text = data['xs']
        labels = data['ys']

        with chainer.using_config('train', False), chainer.no_backprop_mode():   # TODO, is dropout off now?
            output, activations = model.deep_knn_predict(text)
            output.to_cpu()
            activations.to_cpu()

            # activations is (num_layers, batch_size, embed_size), make it be (batch_size, num_layers, embed_size)
            #activations = activations.reshape(activations.shape[1], activations.shape[0], activations.shape[2])
            activations = F.expand_dims(activations,axis = 1)


            # for each layer, get a list of the training data indices
            for current_position_in_minibatch, activation in enumerate(activations.data):
                # activation is size (layers, embed_size)
                for ind, layer_act in enumerate(activation): # layer_act is one layer of activations, ind is current layer index
                    training_indices = []
                    knn = tree_list[ind].neighbours(layer_act)
                    for nn in knn:
                        training_indices.append(nn[1])

                pred_labels = []
                for training_data_index in training_indices:  # for all indices, get their label
                    pred_labels.append(label_list[training_data_index])

                most_common,num_most_common = Counter(pred_labels).most_common(1)[0] # get most common label
                curr_label = labels[current_position_in_minibatch][0]

                if most_common == curr_label:
                    n_correct = n_correct + 1
                total = total + 1


                credibility = float(num_most_common) / float(len(training_indices))

                # print crdedibility scores and print out the sentence and all its nearest neighbors
                #print(credibility)
                #curr_data_input_sentence = ""
                #for input_words in text[current_position_in_minibatch]:
                #    curr_data_input_sentence += idx2word[input_words] + " "
                #print("Test input", curr_data_input_sentence)

                #print("Nearest Neighbors:")
                #for training_data_index in training_indices:
                #    curr_nearest_neighbor_input = train[training_data_index]
                #    curr_nearest_neighbor_input_sentence = ""
                #    for input_words in curr_nearest_neighbor_input[0]:
                #        curr_nearest_neighbor_input_sentence += idx2word[input_words] + " "
                #    print(curr_nearest_neighbor_input_sentence)


    accuracy = float(n_correct) / float(total)
    print('Deep KNN Test Accuracy:{:.04f}'.format(accuracy))

# Unknown questions
# Probably doesn't matter (training accuracy will be near 100%), but use model predicted output or ground truth label?
# 75 neighbors at each layer, or 75 neighbors total?
# concat all hidden layers then count, or have each layer fight it out
# before or after relu?
# Consider using a different distance than euclidean, cosine?
# i am guessing he doesn't include the logits as a "layer" also? I am not doing so right now, see TextClassifier's deep_knn prediction function

if __name__ == '__main__':
    main()
