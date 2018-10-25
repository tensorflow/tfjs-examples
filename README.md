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
    <td><a href="https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py">keras addition_rnn</a></td>
  </tr>
  <tr>
    <td><a href="./baseball-node">baseball-node</a></td>
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
    <td><a href="./boston-housing">boston-housing</a></td>
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
    <td><a href="./cart-pole">cart-pole</a></td>
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
    <td><a href="./custom-layer">custom-layer</a></td>
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
    <td><a href="./iris">iris</a></td>
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
    <td><a href="./lstm-text-generation">lstm-text-generation</a></td>
    <td>Text</td>
    <td>Sequent-to-prediction</td>
    <td>RNN: LSTM</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>IndexedDB</td>
    <td>vega-lite: `fit()` monitoring</td>
    <td><a href="https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py">keras lstm_text_generation]</a></td>
  </tr>
  <tr>
    <td><a href="./mnist">mnist</a></td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
    <td>vega-lite: `fit()` monitoring</td>
    <td><a href="https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py">keras mnist_cnn</a></td>
  </tr>
  <tr>
    <td><a href="./mnist-core">mnist-core</a></td>
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
    <td><a href="./mnist-node">mnist-node</a></td>
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
    <td><a href="./mnist-transfer-cnn">mnist-transfer-cnn</a></td>
    <td>Image</td>
    <td>Multiclass classification (transfer learning)</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
    <td><a href="https://github.com/tensorflow/tfjs-vis">tfjs-vis</a></td>
    <td>
      <a href="https://github.com/keras-team/keras/blob/master/examples/mnist_transfer_cnn.py">
        keras mnist_transfer_cnnn
      </a>
    </td>
  </tr>
  <tr>
    <td><a href="./mobilenet">mobilenet</a></td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
    <td></td>
    <td><a href="https://keras.io/applications/#mobilenet">keras mobilenet application</a></td>
  </tr>
  <tr>
    <td><a href="./polinomial-regression">polinomial-regression</a></td>
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
    <td><a href="./polinomial-regression-core">polinomial-regression-core</a></td>
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
    <td><a href="./sentiment">sentiment</a></td>
    <td>Text</td>
    <td>Sequent-to-regression</td>
    <td>LSTM, 1D convnet</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading model converted from Keras</td>
    <td></td>
    <td>
      <a href="https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py">
        keras imdb_bidirectional_lstm
      </a>
      <a href="https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py">
        keras imdb_cnn
      </a>
    </td>
  </tr>
  <tr>
    <td><a href="./translation">translation</a></td>
    <td>Text</td>
    <td>Sequence-to-sequence</td>
    <td>LSTM encoder and decoder</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading model converted from Keras</td>
    <td></td>
    <td><a href="https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py">keras lstm_seq2seq</a></td>
  </tr>
  <tr>
    <td><a href="./tsne-mnist-canvas">tsne-mnist-canvas</a></td>
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
    <td><a href="./webcam-transfer-learning">webcam-transfer-learning</a></td>
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
    <td><a href="./website-phishing">website-phishing</a></td>
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
