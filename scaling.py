#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
import cupy as cp

import chainer
import chainer.functions as F


from nlp_utils import convert_seq, convert_snli_seq
from utils import setup_model


class TemperatureScaler(chainer.Link):

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        with self.init_scope():
            self.temperature = chainer.Parameter(
                    np.asarray([1.5], dtype=np.float32))
            # self.temperature = chainer.Parameter(
            #         chainer.initializers.One(), (1,))

    def __call__(self, logits):
        return logits / F.broadcast_to(self.temperature, logits.shape)


class ScaledModel(chainer.Chain):

    def __init__(self, model):
        super(ScaledModel, self).__init__()
        with self.init_scope():
            self.model = model
            self.temperature = TemperatureScaler()
        if model.xp == cp:
            self.temperature.to_gpu()

    def predict(self, xs):
        return self.temperature(self.model.predict(xs, no_dropout=True))


class ECELoss:

    def __init__(self, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins+1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, logits, labels):
        # logits = sm.temperature(all_logits)
        scores = cp.asnumpy(F.softmax(logits).data)
        labels = cp.asnumpy(labels.data)
        predictions = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        accuracies = (predictions == labels)
        total_count = accuracies.shape[0]
        ece = 0
        for lower, upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = np.greater(confidences, lower) * \
                     np.less_equal(confidences, upper)
            count = in_bin.sum()
            if count > 0:
                conf_in_bin = (confidences * in_bin).sum() / count
                acc_in_bin = (accuracies * in_bin).sum() / count
                ece += np.abs(conf_in_bin - acc_in_bin) * count / total_count
        return ece


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
    if setup['dataset'] == 'snli':
        converter = convert_snli_seq
        use_snli = True
    else:
        converter = convert_seq
        use_snli = False

    with open(os.path.join(setup['save_path'], 'calib.json')) as f:
        calibration_idx = json.load(f)

    calibration = [train[i] for i in calibration_idx]
    train = [x for i, x in enumerate(train) if i not in calibration_idx]

    sm = ScaledModel(model)

    optim = chainer.optimizers.Adam()
    optim.setup(sm.temperature)

    calib_iter = chainer.iterators.SerialIterator(
            calibration, setup['batchsize'], repeat=False)
    eceloss = ECELoss()

    for i in range(50):
        calib_iter.reset()
        all_logits = []
        all_labels = []
        for i, batch in enumerate(calib_iter):
            batch = converter(batch, device=args.gpu, with_label=True)
            logits = sm.predict(batch['xs'])
            labels = F.concat(batch['ys'], axis=0)
            all_logits.append(logits.data)
            all_labels.append(labels.data)
        all_logits = F.concat(all_logits, axis=0)
        all_labels = F.concat(all_labels, axis=0)

        print(sm.temperature.temperature.data[0],
              eceloss(all_logits, all_labels))

        logits = sm.temperature(all_logits)
        loss = F.softmax_cross_entropy(logits, all_labels)
        sm.temperature.zerograds()
        loss.backward()
        optim.update()


if __name__ == '__main__':
    main()
