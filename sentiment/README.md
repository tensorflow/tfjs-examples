# TensorFlow.js Example: Sentiment Analysis

This demo shows how to perform text sentiment analysis on text using the Layers
API of TensorFlow.js.

It demonstrates loading a pretrained model hosted at a URL, using
`tf.loadLayersModel()`.

Two model variants are provided (CNN and LSTM).  These were trained on a set of
25,000 movie reviews from IMDB, labelled as having positive or negative
sentiment.  This dataset is
[provided by Python Keras](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification),
and the models were trained in Keras as well, based on the
[imdb_cnn](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)
and
[imdb_lstm](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)
examples.

To launch the demo, do

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/sentiment/dist/index.html)

## Training your own model in tfjs-node

To train the model using tfjs-node, do

```sh
yarn
yarn train <MODEL_TYPE>
```

where `MODEL_TYPE` is a required argument that specifies what type of model is to be
trained. The available options are:

- `multihot`: A model that takes a multi-hot encoding of the words in the sequence.
  In terms of data representation and model complexity, this is the simplest model
  in this example.
- `flatten`: A model that flattens the embedding vectors of all words in the sequence.
- `cnn`: A 1D convolutional model, with a dropout layer included.
- `simpleRNN`: A model that uses a SimpleRNN layer (`tf.layers.simpleRNN`)
- `lstm`: A model that uses a LSTM laayer (`tf.layers.lstm`)
- `bidirectionalLSTM`: A model that uses a bidirectional LSTM layer
  (`tf.layers.bidirectional` and `tf.layers.lstm`)

By default, the training happens on the CPU using the Eigen kernels from tfjs-node.
You can make the training happen on GPU by adding the `--gpu` flag to the command, e.g.,

```sh
yarn train --gpu <MODEL_TYPE>
```

The training process will download the training data and metadata form the web
if they haven't been downloaded before. After the model training completes, the model
will be saved to the `dist/resources` folder, alongside a `metadata.json` file.
Then when you run `yarn watch`, you will see a "Load local model" button in the web
page, which allows you to use the locally-trained model for inference in the browser.

Other arguments of the `yarn train` command include:

- `--maxLen` allows you to specify the sequence length.
- `--numWords` allows you to specify the vocabulary size.
- `--embeddingSize` allows you to adjust the dimensionality of the embedding vectors.
- `--epochs`, `--batchSize`, and `--validationSplit` are training-related settings.
- `--modelSavePath` allows you to specify where to store the model and metadata after
  training completes.
- `--embeddingFilesPrefix` Prefix for the path to which to save the embedding vectors
  and labels files (optinal). See the section below for details.

The detailed code for training are in the file [train.js](./train.js).

### Visualizing the word embeddings in embedding projector

If you train a word embedding-based model (e.g., `cnn` or `lstm`), you can let the
`yarn train` script write the embedding vectors, together with the corresponding
word labels, to files after the model training completes. This is done using the
``--embeddingFilesPrefix`, e.g.,

```sh
yarn train --maxLen 500 cnn --epochs 2 --embeddingFilesPrefix /tmp/imdb_embed
```

The above command will generate two files:

- `/tmp/imdb_embed_vectors.tsv`: A tab-separated-values file that for the numeric
  values of the word embeddings. Each line contains the embedding vector from a
  word.
- `/tmp/imdb_embed_labels.tsv`: A file consisting of the word labels that correspond
  to the vectors in the previous file. Each line is a word.

These files can be directly uploaded to the Embedding Projector
(https://projector.tensorflow.org/) for visualization using the
[T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) or
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) algorithm

See example screenshot:
![image](https://user-images.githubusercontent.com/16824702/52145038-f0fce480-262d-11e9-9313-9a5014ace25f.png)

### Running unit tests

This example comes with unit tests. If you would like to submit changes to the code, 
be sure to run the tests and ensure they pass first:

```sh
yarn
yarn test
```
