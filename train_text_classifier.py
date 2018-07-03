#!/usr/bin/env python
import argparse
import datetime
import json
import os
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import nets
from nlp_utils import convert_seq, convert_snli_seq
import text_datasets


def create_parser():
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--combine_snli', action='store_true')
    parser.add_argument('--dataset', '-data', default='TREC',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj',
                                 'snli'],
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
    current_datetime = '{}'.format(datetime.datetime.today())

    # Load a dataset
    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            char_based=args.char_based)
    elif args.dataset == 'snli':
        train, test, vocab = text_datasets.get_snli(
            char_based=args.char_based, combine=args.combine_snli)
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
    if args.dataset == 'snli':
        n_class = 3
    else:
        n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Save vocabulary and model's setting
    current = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
            current,
            args.out,
            '{}_{}'.format(args.dataset, args.model)
            )
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    vocab_path = os.path.join(save_path, 'vocab.json')
    model_path = os.path.join(save_path, 'best_model.npz')
    setup_path = os.path.join(save_path, 'args.json')

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)

    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(setup_path, 'w') as f:
        json.dump(model_setup, f)
    print(json.dumps(model_setup, indent=2))

    # Setup a model
    if args.model == 'rnn':
        Encoder = nets.RNNEncoder
    if args.model == 'bilstm':
        Encoder = nets.BiLSTMEncoder
    elif args.model == 'cnn':
        Encoder = nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)
    if args.dataset == 'snli':
        model = nets.SNLIClassifier(encoder, combine=args.combine_snli)
    else:
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
    if args.dataset == 'snli' and not args.combine_snli:
        converter = convert_snli_seq
    else:
        converter = convert_seq

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        converter=converter, device=args.gpu)
    trainer = training.Trainer(
            updater, (args.epoch, 'epoch'),
            out=os.path.join(
                args.out, '{}_{}'.format(args.dataset, args.model)))

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        test_iter, model,
        converter=converter, device=args.gpu))

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

    idx2word = {}   # build reverse dict
    for word, idx in vocab.items():
        idx2word[idx] = word

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
