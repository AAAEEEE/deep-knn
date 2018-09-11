# Deep k-Nearest Neighbors and Interpretable NLP

This is the official code for the 2018 EMNLP Interpretability Workshop paper, [Interpreting Neural Networks with Nearest Neighbors](https://arxiv.org/abs/1809.02847). 

This repository contains the code for:

* Deep k-Nearest Neighbors for text classification models. Allows pretrained word vectors, character level models, etc. on a number of datasets
* Saliency map techniques for NLP, such as leave one out and gradient. Also includes our conformity leave one out method.
* Create visualizations like the ones on our paper's [supplementary website](https://sites.google.com/view/language-dknn/).
* Temperature scaling as described in [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
* SNLI interpretations

## Dependencies

This code is written in python using the highly underrated Chainer framework. If you know PyTorch, you will love it =).

Dependencies include:

* Python 2/3
* [Chainer](https://chainer.org/)
* tqdm
* numpy

If you want to do efficient nearest neighbor lookup:
* Scikit-Learn (for KDTree)
* nearpy (for locally sensitive hashing)

If you want to visualize saliency maps:
* matplotlib


This code is built off Chainers [text classification example](https://github.com/chainer/chainer/tree/master/examples/text_classification). See their documentation and code to understand the basic layout of our project. 

## Files


To train a model:  
```
python train_text_classifier.py --dataset stsa.binary --model cnn
```
The output directory `result` contains:  
- `best_model.npz`: a model snapshot, which won the best accuracy for validation data during training
- `vocab.json`: model's vocabulary dictionary as a json file
- `args.json`: model's setup as a json file, which also contains paths of the model and vocabulary
- `calib.json`: The indices of the held out training data that will be used to calibrate the DkNN model

To run a model with and without DkNN:  
```
python run_dknn.py --model-setup results/DATASET_MODEL/args.json
```

- Where `results/DATASET_MODEL/args.json` is the argument log that is generated after training a model
- This command will store the activations for all of the training data into a KDTree, calibrate the credibility values, and run the model with and without DkNN.  

## Word Vectors

In our paper, we used GloVe word vectors, though any pretrained vectors should work fine (word2vec, fastText, etc.). To obtain GloVe vectors, run the following commands.

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

Then pass the pretrained vectors in using the argument `--word_vectors glove.840B.300d.txt` when training a model using `train_text_classifier.py`

## Temperature Scaling

`scaling.py` contains the temperature scaling implementation.

## Interpretations and Visualizations

All of the code for generating interpretations using leave one out (conformity, confidence, or calibrated confidence) and first-order gradient is contained in `interpretations.py`. See the code for details on running with the desired settings. You should first train a model (see above), and then pass that in.

The code for visualization is also present in `interpretations.py`.

## References

Please consider citing [1](#dknn-language) if you found this code or our work beneficial to your research.

### Interpreting Neural Networks with Nearest Neighbors

[1] Eric Wallace, Shi Feng, and Jordan Boyd-Graber, [Interpreting Neural Networks with Nearest Neighbors](https://arxiv.org/abs/1809.02847). 

```
@article{Wallace2018Neighbors,
  title={Interpreting Neural Networks with Nearest Neighbors},
  author={Eric Wallace and Shi Feng and Jordan Boyd-Graber},
  journal={EMNLP 2018 Workshop on Analyzing and Interpreting Neural Networks for NLP},  
  year={2018},  
}
```

## Contact

For issues with code or suggested improvements, feel free to open a pull request.

To contact the authors, reach out to Eric Wallace (ewallac2@umd.edu) and Shi Feng (shifeng@cs.umd.edu).
