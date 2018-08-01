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
import {IMAGE_H, IMAGE_W, MnistData} from './data';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from './ui';

function createConvModel(includeDropout) {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  const model = tf.sequential();

  // The first layer of the convolutional neural network plays a dual role:
  // it is both the input layer of the neural network and a layer that performs
  // the first convolution operation on the input. It receives the 28x28 pixels
  // black and white images. This input layer uses 8 filters with a kernel size
  // of 5 pixels each. It uses a simple RELU activation function which pretty
  // much just looks like this: __/
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu'
  }));

  // After the first layer we include a MaxPooling layer. This acts as a sort of
  // downsampling using max values in a region instead of averaging.
  // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Our third layer is another convolution, this time with 16 filters.
  model.add(tf.layers.conv2d(
      {kernelSize: 5, filters: 16, strides: 1, activation: 'relu'}));

  // Max pooling again.
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten({}));

  if (includeDropout) {
    model.add(tf.layers.dropout({rate: 0.25}));
  }

  model.add(tf.layers.dense({units: 100, activation: 'relu'}));

  if (includeDropout) {
    model.add(tf.layers.dropout({rate: 0.5}));
  }

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
  // represent numbers, but it's the same idea if you had classes that
  // represented other entities like dogs and cats (two output classes: 0, 1).
  // We use the softmax function as the activation for the output layer as it
  // creates a probability distribution over our 10 classes so their output
  // values sum to 1.
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}

function createDenseModel(includeDropout) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  if (includeDropout) {
    model.add(tf.layers.dropout({rate: 0.25}));
  }
  model.add(tf.layers.dense({units: 40, activation: 'relu'}));
  if (includeDropout) {
    model.add(tf.layers.dropout({rate: 0.5}));
  }
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

async function train(model) {
  ui.isTraining();

  // Now that we've defined our model, we will define our optimizer. The
  // optimizer will be used to optimize our model's weight values during
  // training so that we can decrease our training loss and increase our
  // classification accuracy.

  // The learning rate defines the magnitude by which we update our weights each
  // training step. The higher the value, the faster our loss values converge,
  // but also the more likely we are to overshoot optimal parameters
  // when making an update. A learning rate that is too low will take too long
  // to find optimal (or good enough) weight parameters while a learning rate
  // that is too high may overshoot optimal parameters. Learning rate is one of
  // the most important hyperparameters to set correctly. Finding the right
  // value takes practice and is often best found empirically by trying many
  // values.
  const LEARNING_RATE = 0.01;

  // We are using Stochastic Gradient Descent (SGD) as our optimization
  // algorithm. This is the most famous modern optimization algorithm in deep
  // learning and it is largely to thank for the current machine learning
  // renaissance. Most other optimizers you will come across (e.g. ADAM,
  // RMSProp, AdaGrad, Momentum) are variants on SGD. SGD is an iterative method
  // for minimizing an objective function. It tries to find the minimum of our
  // loss function with respect to the model's weight parameters.
  const optimizer = tf.train.rmsprop(LEARNING_RATE);

  // We compile our model by specifying an optimizer, a loss function, and a
  // list of metrics that we will use for model evaluation. Here we're using a
  // categorical crossentropy loss, the standard choice for a multi-class
  // classification problem like MNIST digits.
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Batch size is another important hyperparameter. It defines the number of
  // examples we group together, or batch, between updates to the model's
  // weights during training. A value that is too low will update weights using
  // too few examples and will not generalize well. Larger batch sizes require
  // more memory resources and aren't guaranteed to perform better.
  const BATCH_SIZE = 512;

  const TRAIN_EPOCHS = 3;

  // We'll keep a buffer of loss and accuracy values over time.
  let trainBatchCount = 0;
  const lossValues = [];
  const accuracyValues = [];

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
      Math.ceil(trainData.xs.shape[0] / BATCH_SIZE) * TRAIN_EPOCHS;

  await model.fit(trainData.xs, trainData.labels, {
    batchSize: BATCH_SIZE,
    validationSplit: 0.15,
    epochs: TRAIN_EPOCHS,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
            `Training... (` +
            `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
            ` complete)`);
        lossValues.push(
            {'batch': trainBatchCount, 'loss': logs.loss, 'set': 'train'});
        accuracyValues.push(
            {'batch': trainBatchCount, 'accuracy': logs.acc, 'set': 'train'});
        ui.plotLosses(lossValues);
        ui.plotAccuracies(accuracyValues);
      },
      onEpochEnd: async (epoch, logs) => {
        lossValues.push({
          'batch': trainBatchCount,
          'loss': logs.val_loss,
          'set': 'validation'
        });
        accuracyValues.push({
          'batch': trainBatchCount,
          'accuracy': logs.val_acc,
          'set': 'validation'
        });
        console.log(
            `onEpochEnd: epoch = ${epoch}, log = ${JSON.stringify(logs)}`);
        ui.plotLosses(lossValues);
        ui.plotAccuracies(accuracyValues);
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAcc = testResult[1].dataSync()[0];
  ui.logStatus(
      `Final validation accuracy: ` +
      `${
          (accuracyValues[accuracyValues.length - 1].accuracy * 100)
              .toFixed(3)}; ` +
      `Final test accuracy: ${(testAcc * 100).toFixed(3)}`);
}

async function showPredictions() {
  const testExamples = 100;
  const batch = data.nextTestBatch(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const output = model.predict(batch.xs.reshape([-1, IMAGE_H, IMAGE_W, 1]));

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

function createModel() {
  let model;
  const modelType = ui.getModelTypeId();
  console.log(`modelType = ${modelType}`);  // DEBUG
  if (modelType === 'ConvNet with Dropout') {
    model = createConvModel(true);
  } else if (modelType === 'ConvNet without Dropouot') {
    model = createConvModel(false);
  } else if (modelType === 'DenseNet with Dropouot') {
    model = createDenseModel(true);
  } else if (modelType === 'DenseNet without Dropout') {
    model = createDenseModel(false);
  } else {
    throw new Error(`Invalid model type: ${modelType}`);
  }
  return model;
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = createModel();
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model);

  showPredictions();
});
