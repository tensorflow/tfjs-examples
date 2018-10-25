# TensorFlow.js Examples

This repository contains a set of examples implemented in
[TensorFlow.js](http://js.tensorflow.org).

Each example directory is standalone so the directory can be copied
to another project.

# Overview of Examples

<table>
  <tr>
    <th>Example name</th>
    <th>Input data type</th>
    <th>Task type</th>
    <th>Model type</th>
    <th>Training</th>
    <th>Inference</th>
    <th>API type</th>
    <th>Save-load operations</th>
    <th>Model visualization</th>
    <th>Python source / analog</th>
  <tr>
    <td><a href="./addition-rnn">addition-rnn</a></td>
    <td>Text</td>
    <td>Sequence-to-sequence</td>
    <td>RNN: SimpleRNN, GRU and LSTM</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td>vega-lite: `fit()` monitoring</td>
    <td>[keras addition_rnn](https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py)</td>
  </tr>
  <tr>
    <td>[baseball-node](./baseball-node)</td>
    <td>Numeric</td>
    <td>Multiclass classification</td>
    <td>Multilayer perceptron</td>
    <td>Node.js</td>
    <td>Node.js</td>
    <td>Layers</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[boston-housing](./boston-housing)</td>
    <td>Numeric</td>
    <td>Regression</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td>vega-lite: `fit()` monitoring</td>
    <td></td>
  </tr>
  <tr>
    <td>[cart-pole](./cart-pole)</td>
    <td></td>
    <td>Reinforcement learning</td>
    <td>Policy gradient</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>IndexedDB</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[custom-layer](./custom-layer)</td>
    <td></td>
    <td>(Illustrates how to define and use a custom Layer subtype)</td>
    <td></td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[iris](./iris)</td>
    <td>Numeric</td>
    <td>Multiclass classification</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>IndexedDB; Loading model converted from Kras</td>
    <td>vega-lite: `fit()` monitoring</td>
    <td></td>
  </tr>
  <tr>
    <td>[lstm-text-generation](./lstm-text-generation)</td>
    <td>Text</td>
    <td>Sequent-to-prediction</td>
    <td>RNN: LSTM</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>IndexedDB</td>
    <td>vega-lite: `fit()` monitoring</td>
    <td>[keras lstm_text_generation](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)</td>
  </tr>
  <tr>
    <td>[mnist](./mnist)</td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td>vega-lite: `fit()` monitoring</td>
    <td>[keras mnist_cnn](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)</td>
  </tr>
  <tr>
    <td>[mnist-core](./mnist-core)</td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Core (Ops)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[mnist-node](./mnist-node)</td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Node.js</td>
    <td>Node.js</td>
    <td>Layers</td>
    <td>Saving to filesystem</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[mnist-transfer-cnn](./mnist-transfer-cnn)</td>
    <td>Image</td>
    <td>Multiclass classification (transfer learning)</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
    <td>[tfjs-vis](https://github.com/tensorflow/tfjs-vis)</td>
    <td>[keras mnist_transfer_cnn](https://github.com/keras-team/keras/blob/master/examples/mnist_transfer_cnn.py)</td>
  </tr>
  <tr>
    <td>[mobilenet](./mobilenet)</td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
    <td></td>
    <td>[keras mobilenet](https://keras.io/applications/#mobilenet)</td>
  </tr>
  <tr>
    <td>[polynomial-regression](./polynomial-regression)</td>
    <td>Numeric</td>
    <td>Regression</td>
    <td>Shallow neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[polynomial-regression-core](./polynomial-regression-core)</td>
    <td>Numeric</td>
    <td>Regression</td>
    <td>Shallow neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Core (Ops)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[sentiment](./sentiment)</td>
    <td>Text</td>
    <td>Sequent-to-regression</td>
    <td>LSTM, 1D convnet</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading model converted from Keras</td>
    <td></td>
    <td>[keras imdb_bidirectional_lstm](https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py), [keras imdb_cnn](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)</td>
  </tr>
  <tr>
    <td>[translation](./translation)</td>
    <td>Text</td>
    <td>Sequence-to-sequence</td>
    <td>LSTM encoder and decoder</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading model converted from Keras</td>
    <td></td>
    <td>[keras lstm_seq2seq](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)</td>
  </tr>
  <tr>
    <td>[tsne-mnist-canvas](./tsne-mnist-canvas)</td>
    <td></td>
    <td>Dimension reduction and data visualization</td>
    <td>tSNE</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Core (Ops)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[webcam-transfer-learning](./webcam-transfer-learning)</td>
    <td>Image</td>
    <td>Multiclass classification (transfer learning)</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[website-phishing](./website-phishing)</td>
    <td>Numeric</td>
    <td>Binary classification</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td>vega-lite: `fit()` monitoring</td>
    <td></td>
  </tr>
</table>

# Dependencies

Except for `getting_started`, all the examples require the following dependencies to be installed.

 - Node.js version 8.9 or higher
 - [NPM cli](https://docs.npmjs.com/cli/npm) OR [Yarn](https://yarnpkg.com/en/)

## How to build an example
`cd` into the directory

If you are using `yarn`:

```sh
cd mnist-core
yarn
yarn watch
```

If you are using `npm`:
```sh
cd mnist-core
npm install
npm run watch
```

### Details

The convention is that each example contains two scripts:

- `yarn watch` or `npm run watch`: starts a local development HTTP server which watches the
filesystem for changes so you can edit the code (JS or HTML) and see changes when you refresh the page immediately.

- `yarn build` or `npm run build`: generates a `dist/` folder which contains the build artifacts and
can be used for deployment.

## Contributing

If you want to contribute an example, please reach out to us on
[Github issues](https://github.com/tensorflow/tfjs-examples/issues)
before sending us a pull request as we are trying to keep this set of examples
small and highly curated.
