# TensorFlow.js Examples

This repository contains a set of examples implemented in
[TensorFlow.js](http://js.tensorflow.org).

Each example directory is standalone so the directory can be copied
to another project.

# Overview of Examples

<table>
  <tr>
    <th>Example name</th>
    <th>Demo link</th>
    <th>Input data type</th>
    <th>Task type</th>
    <th>Model type</th>
    <th>Training</th>
    <th>Inference</th>
    <th>API type</th>
    <th>Save-load operations</th>
  <tr>
    <td><a href="./abalone-node">abalone-node</a></td>
    <td></td>
    <td>Numeric</td>
    <td>Loading data from local file and training in Node.js</td>
    <td>Multilayer perceptron</td>
    <td>Node.js</td>
    <td>Node.js</td>
    <td>Layers</td>
    <td>Saving to filesystem and loading in Node.js</td>
  </tr>
  <tr>
    <td><a href="./addition-rnn">addition-rnn</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/addition-rnn/dist/index.html">🔗</a></td>
    <td>Text</td>
    <td>Sequence-to-sequence</td>
    <td>RNN: SimpleRNN, GRU and LSTM</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./addition-rnn-webworker">addition-rnn-webworker</a></td>
    <td></td>
    <td>Text</td>
    <td>Sequence-to-sequence</td>
    <td>RNN: SimpleRNN, GRU and LSTM</td>
    <td>Browser: Web Worker</td>
    <td>Browser: Web Worker</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./baseball-node">baseball-node</a></td>
    <td></td>
    <td>Numeric</td>
    <td>Multiclass classification</td>
    <td>Multilayer perceptron</td>
    <td>Node.js</td>
    <td>Node.js</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./boston-housing">boston-housing</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/boston-housing/dist/index.html">🔗</a></td>
    <td>Numeric</td>
    <td>Regression</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./cart-pole">cart-pole</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/cart-pole/dist/index.html">🔗</a></td>
    <td></td>
    <td>Reinforcement learning</td>
    <td>Policy gradient</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>IndexedDB</td>
  </tr>
  <tr>
    <td><a href="./chrome-extension">chrome-extension</a></td>
    <td></td>
    <td>Image</td>
    <td>(Deploying TF.js in Chrome extension)</td>
    <td>Convnet</td>
    <td></td>
    <td>Browser</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./custom-layer">custom-layer</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/custom-layer/dist/index.html">🔗</a></td>
    <td></td>
    <td>(Defining a custom Layer subtype)</td>
    <td></td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./data-csv">data-csv</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/data-csv/dist/index.html">🔗</a></td>
    <td></td>
    <td>Building a tf.data.Dataset from a remote CSV</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./data-generator">data-generator</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/data-generator/dist/index.html">🔗</a></td>
    <td></td>
    <td>Building a tf.data.Dataset using a generator</td>
    <td>Regression</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./date-conversion-attention">date-conversion-attention</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/date-conversion-attention/dist/index.html">🔗</a></td>
    <td>Text</td>
    <td>Text-to-text conversion</td>
    <td>Attention mechanism, RNN</td>
    <td>Node.js</td>
    <td>Browser and Node.js</td>
    <td>Layers</td>
    <td>Saving to filesystem and loading in browser</td>
  </tr>
  <tr>
    <td><a href="./electron">electron</a></td>
    <td></td>
    <td>Image</td>
    <td>(Deploying TF.js in Electron-based desktop apps)</td>
    <td>Convnet</td>
    <td></td>
    <td>Node.js</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./fashion-mnist-vae">fashion-mnist-vae</a></td>
    <td></td>
    <td>Image</td>
    <td>Generative</td>
    <td>Variational autoencoder (VAE)</td>
    <td>Node.js</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Export trained model from tfjs-node and load it in browser</td>
  </tr>
  <tr>
    <td><a href="./interactive-visualizers">interactive-visualizers</a></td>
    <td></td>
    <td>Image</td>
    <td>Multiclass classification, object detection, segmentation</td>
    <td></td>
    <td></td>
    <td>Browser</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./iris">iris</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/iris/dist/index.html">🔗</a></td>
    <td>Numeric</td>
    <td>Multiclass classification</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./iris-fitDataset">iris-fitDataset</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/iris-fitDataset/dist/index.html">🔗</a></td>
    <td>Numeric</td>
    <td>Multiclass classification</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./jena-weather">jena-weather</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/jena-weather/dist/index.html">🔗</a></td>
    <td>Sequence</td>
    <td>Sequence-to-prediction</td>
    <td>MLP and RNNs</td>
    <td>Browser and Node</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./lstm-text-generation">lstm-text-generation</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/lstm-text-generation/dist/index.html">🔗</a></td>
    <td>Text</td>
    <td>Sequence prediction</td>
    <td>RNN: LSTM</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>IndexedDB</td>
  </tr>
  <tr>
    <td><a href="./mnist">mnist</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/mnist/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./mnist-acgan">mnist-acgan</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/mnist-acgan/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Generative Adversarial Network (GAN)</td>
    <td>Convolutional neural network; GAN</td>
    <td>Node.js</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Saving to filesystem from Node.js and loading it in the browser</td>
  </tr>
  <tr>
    <td><a href="./mnist-core">mnist-core</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/mnist-core/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Core (Ops)</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./mnist-node">mnist-node</a></td>
    <td></td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td>Node.js</td>
    <td>Node.js</td>
    <td>Layers</td>
    <td>Saving to filesystem</td>
  </tr>
  <tr>
    <td><a href="./mnist-transfer-cnn">mnist-transfer-cnn</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/mnist-transfer-cnn/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Multiclass classification (transfer learning)</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
  </tr>
  <tr>
    <td><a href="./mobilenet">mobilenet</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/mobilenet/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Multiclass classification</td>
    <td>Convolutional neural network</td>
    <td></td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
  </tr>
  <tr>
    <td><a href="./polynomial-regression">polynomial-regression</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/polynomial-regression/dist/index.html">🔗</a></td>
    <td>Numeric</td>
    <td>Regression</td>
    <td>Shallow neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./polynomial-regression-core">polynomial-regression-core</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/polynomial-regression-core/dist/index.html">🔗</a></td>
    <td>Numeric</td>
    <td>Regression</td>
    <td>Shallow neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Core (Ops)</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./quantization">quantization</a></td>
    <td></td>
    <td>Various</td>
    <td>Demonstrates the effect of post-training weight quantization</td>
    <td>Various</td>
    <td>Node.js</td>
    <td>Node.js</td>
    <td>Layers</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./sentiment">sentiment</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/sentiment/dist/index.html">🔗</a></td>
    <td>Text</td>
    <td>Sequence-to-binary-prediction</td>
    <td>LSTM, 1D convnet</td>
    <td>Node.js or Python</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Load model from Keras and tfjs-node</td>
  </tr>
  <tr>
    <td><a href="./simple-object-detection">simple-object-detection</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Object detection</td>
    <td>Convolutional neural network (transfer learning)</td>
    <td>Node.js</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Export trained model from tfjs-node and load it in browser</td>
  </tr>
  <tr>
    <td><a href="./snake-dqn">snake-dqn</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/snake-dqn/index.html">🔗</a></td>
    <td></td>
    <td>Reinforcement learning</td>
    <td>Deep Q-Network (DQN)</td>
    <td>Node.js</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Export trained model from tfjs-node and load it in browser</td>
  </tr>
  <tr>
    <td><a href="./translation">translation</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/translation/dist/index.html">🔗</a></td>
    <td>Text</td>
    <td>Sequence-to-sequence</td>
    <td>LSTM encoder and decoder</td>
    <td>Node.js or Python</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Load model converted from Keras</td>
  </tr>
  <tr>
    <td><a href="./tsne-mnist-canvas">tsne-mnist-canvas</a></td>
    <td></td>
    <td></td>
    <td>Dimension reduction and data visualization</td>
    <td>tSNE</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Core (Ops)</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="./webcam-transfer-learning">webcam-transfer-learning</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html">🔗</a></td>
    <td>Image</td>
    <td>Multiclass classification (transfer learning)</td>
    <td>Convolutional neural network</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
    <td>Loading pretrained model</td>
  </tr>
  <tr>
    <td><a href="./website-phishing">website-phishing</a></td>
    <td><a href="https://storage.googleapis.com/tfjs-examples/website-phishing/dist/index.html">🔗</a></td>
    <td>Numeric</td>
    <td>Binary classification</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
    <td>Layers</td>
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
[Github issues](https://github.com/tensorflow/tfjs/issues)
before sending us a pull request as we are trying to keep this set of examples
small and highly curated.

### Running Presubmit Tests

Before you send a pull request, it is a good idea to run the presubmit tests
and make sure they all pass. To do that, execute the following commands in the
root directory of tfjs-examples:

```sh
yarn
yarn presubmit
```

The `yarn presubmit` command executes the unit tests and lint checks of all
the exapmles that contain the `yarn test` and/or `yarn lint` scripts. You
may also run the tests for individual exampls by cd'ing into their respective
subdirectory and executing `yarn`, followed by `yarn test` and/or `yarn lint`.
