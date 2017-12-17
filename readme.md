# Graph based dependency parsing

This repository replicates research by ([Kiperwasser and Golberg, 2016](https://arxiv.org/abs/1603.04351)). 
Datasets are taken from the [Universal dependencies project](http://universaldependencies.org/).
Initial word embeddings were obtained with the [GloVe algorithm](https://nlp.stanford.edu/projects/glove/) and can be found in the embeddings subfolder of each language folder.

## Usage

Models are trained with the [NLP_training.py](./NLP_training.py) script.
It takes the following arguments: hidden size for the arc-prediction MLP, learning rate, hidden size for the label-prediction MLP, language (must correspond to a language directory, 'en' and 'de' are currently available), number of epochs and modelname (a model will be saved after each epoch).

```
python NLP_training.py 25 0.0001 50 de 2 model1
```

Evaluation is done with the [test.py](./test.py) script. First four parameters are equivalent to the training script (although learning rate won't be used), and aditionally the dataset that shall be used for evaluation (train, dev or test) and modelname.

```
python NLP_training.py 25 0.0001 50 de test model1_e16
```

The script will print UAS and LAS. It will also write to a file named `conllu` which can be evaluated with the [official evaluation script](http://universaldependencies.org/conll17/evaluation.html).

Some pretrained models achieving UAS scores around 70 % can be found in the models folder.