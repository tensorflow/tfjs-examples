/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

// This is a helper class for loading and managing MNIST data specifically.
// It is a useful example of how you could create your own data manager class
// for arbitrary data though. It's worth a look :)
import {MnistData} from './data';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from './ui';

// Create a sequential neural network model. tf.sequential provides an API for
// creating "stacked" models where the output from one layer is used as the
// input to the next layer.
const model = tf.sequential();

// The first layer of the convolutional neural network plays a dual role:
// it is both the input layer of the neural network and a layer that performs
// the first convolution operation on the input. It receives the 28x28 pixels
// black and white images. This input layer uses 8 filters with a kernel size
// of 5 pixels each. It uses a simple RELU activation function which pretty
// much just looks like this: __/
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));

// After the first layer we include a MaxPooling layer. This acts as a sort of
// downsampling using max values in a region instead of averaging.
// https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

// Our third layer is another convolution, this time with 16 filters.
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));

// Max pooling again.
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

// Now we flatten the output from the 2D filters into a 1D vector to prepare
// it for input into our last layer. This is common practice when feeding
// higher dimensional data to a final classification output layer.
model.add(tf.layers.flatten());

// Our last layer is a dense layer which has 10 output units, one for each
// output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
// represent numbers, but it's the same idea if you had classes that represented
// other entities like dogs and cats (two output classes: 0, 1).
// We use the softmax function as the activation for the output layer as it
// creates a probability distribution over our 10 classes so their output values
// sum to 1.
model.add(tf.layers.dense(
    {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

// Now that we've defined our model, we will define our optimizer. The optimizer
// will be used to optimize our model's weight values during training so that
// we can decrease our training loss and increase our classification accuracy.

// The learning rate defines the magnitude by which we update our weights each
// training step. The higher the value, the faster our loss values converge,
// but also the more likely we are to overshoot optimal parameters
// when making an update. A learning rate that is too low will take too long to
// find optimal (or good enough) weight parameters while a learning rate that is
// too high may overshoot optimal parameters. Learning rate is one of the most
// important hyperparameters to set correctly. Finding the right value takes
// practice and is often best found empirically by trying many values.
const LEARNING_RATE = 0.15;

// We are using Stochastic Gradient Descent (SGD) as our optimization algorithm.
// This is the most famous modern optimization algorithm in deep learning and
// it is largely to thank for the current machine learning renaissance.
// Most other optimizers you will come across (e.g. ADAM, RMSProp, AdaGrad,
// Momentum) are variants on SGD. SGD is an iterative method for minimizing an
// objective function. It tries to find the minimum of our loss function with
// respect to the model's weight parameters.
const optimizer = tf.train.sgd(LEARNING_RATE);

// We compile our model by specifying an optimizer, a loss function, and a list
// of metrics that we will use for model evaluation. Here we're using a
// categorical crossentropy loss, the standard choice for a multi-class
// classification problem like MNIST digits.
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

// Batch size is another important hyperparameter. It defines the number of
// examples we group together, or batch, between updates to the model's weights
// during training. A value that is too low will update weights using too few
// examples and will not generalize well. Larger batch sizes require more memory
// resources and aren't guaranteed to perform better.
const BATCH_SIZE = 64;

// The number of batches to train on before freezing the model and considering
// it trained. This will result in BATCH_SIZE x TRAIN_BATCHES examples being
// fed to the model during training.
const TRAIN_BATCHES = 150;

// Every few batches, test accuracy over many examples. Ideally, we'd compute
// accuracy over the whole test set, but for performance we'll use a subset.

// The number of test examples to predict each time we test. Because we don't
// update model weights during testing this value doesn't affect model training.
const TEST_BATCH_SIZE = 1000;
// The number of training batches we will run between each test batch.
const TEST_ITERATION_FREQUENCY = 5;

async function train() {
  ui.isTraining();

  // We'll keep a buffer of loss and accuracy values over time.
  const lossValues = [];
  const accuracyValues = [];

  // Iteratively train our model on mini-batches of data.
  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const [batch, validationData] = tf.tidy(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);

      let validationData;
      // Every few batches test the accuracy of the model.
      if (i % TEST_ITERATION_FREQUENCY === 0) {
        const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
        validationData = [
          // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
          // that we can feed it to our convolutional neural net.
          testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
        ];
      }
      return [batch, validationData];
    });

    // The entire dataset doesn't fit into memory so we call train repeatedly
    // with batches using the fit() method.
    const history = await model.fit(
        batch.xs, batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    // Plot loss / accuracy.
    lossValues.push({'batch': i, 'loss': loss, 'set': 'train'});
    ui.plotLosses(lossValues);

    if (validationData != null) {
      accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
      ui.plotAccuracies(accuracyValues);
    }

    // Call dispose on the training/test tensors to free their GPU memory.
    tf.dispose([batch, validationData]);

    // tf.nextFrame() returns a promise that resolves at the next call to
    // requestAnimationFrame(). By awaiting this promise we keep our model
    // training from blocking the main UI thread and freezing the browser.
    await tf.nextFrame();
  }
}

async function showPredictions() {
  const testExamples = 100;
  const batch = data.nextTestBatch(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

    // tf.argMax() returns the indices of the maximum values in the tensor along
    // a specific axis. Categorical classification tasks like this one often
    // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
    // one element for each output class. All values in the vector are 0
    // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
    // output from model.predict() will be a probability distribution, so we use
    // argMax to get the index of the vector element that has the highest
    // probability. This is our prediction.
    // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
    // dataSync() synchronously downloads the tf.tensor values from the GPU so
    // that we can use them in our normal CPU JavaScript code
    // (for a non-blocking version of this function, use data()).
    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(batch, predictions, labels);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
async function mnist() {
  await load();
  await train();
  showPredictions();
}
mnist();
