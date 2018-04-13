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

import {MnistData} from './data';
import * as ui from './ui';

/**
 * This custom layer is similar to the 'relu' non-linear Activation `Layer`, but
 * it keeps both the negative and positive signal.  The input is centered at the
 * mean value, and then the negative activations and positive activations are
 * separated into different channels, meaning that there are twice as many
 * output channels as input channels.
 *
 * Implementing a custom `Layer` in general invovles specifying a `call`
 * function, and possibly also a `computeOutputShape` and `build` function. This
 * layer does not need a custom `build` function because it does not store any
 * variables.
 *
 * TODO(bileschi): File a github issue for the loading / saving of custom
 * layers.
 */
class Antirectifier extends tf.layers.Layer {
  constructor() {
    super({});
    // TODO(bileschi): Can we point to documentation on masking here?
    this.supportsMasking = true;
  }

  /**
   * This layer only works on 4D Tensors [batch, height, width, channels],
   * and produces output with twice as many channels.
   *
   * layer.computeOutputShapes must be overridden in the case that the output
   * shape is not the same as the input shape.
   * @param {*} inputShapes
   */
  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1], inputShape[2], 2 * inputShape[3]]
  }

  call(inputs, kwargs) {
    this.invokeCallHook(inputs, kwargs);
    const origShape = inputs[0].shape;
    const flatShape =
        [origShape[0], origShape[1] * origShape[2] * origShape[3]];
    const flattened = inputs[0].reshape(flatShape);
    const centered = tf.sub(flattened, flattened.mean(1).expandDims(1));
    const pos = centered.relu().reshape(origShape);
    const neg = centered.neg().relu().reshape(origShape);
    return tf.concat([pos, neg], 3);
  }
}

// Set up base model
const baseModel = tf.sequential();
baseModel.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
baseModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
baseModel.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
baseModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
baseModel.add(tf.layers.flatten());
baseModel.add(tf.layers.dense(
    {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));


// Set up the custom model using new Antirectifier instead of relu activation.
const customModel = tf.sequential();
customModel.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'linear',
  kernelInitializer: 'varianceScaling'
}));
customModel.add(new Antirectifier());
customModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
customModel.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'linear',
  kernelInitializer: 'varianceScaling'
}));
customModel.add(new Antirectifier());
customModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
customModel.add(tf.layers.flatten());
customModel.add(tf.layers.dense(
    {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

// Compile both models.
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
baseModel.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});
customModel.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

// Every few batches, test accuracy over many examples. Ideally, we'd compute
// accuracy over the whole test set, but for performance we'll use a subset.
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

async function train() {
  ui.isTraining();

  const lossValues = [];
  const accuracyValues = [];

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = data.nextTrainBatch(BATCH_SIZE);

    let testBatch;
    let validationData;
    // Every few batches test the accuracy of the mode.
    if (i % TEST_ITERATION_FREQUENCY === 0) {
      testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
      validationData = [
        testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
      ];
    }

    // The entire dataset doesn't fit into memory so we call fit repeatedly
    // with batches.
    const baseHistory = await baseModel.fit(
        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});
    const baseLoss = baseHistory.history.loss[0];
    const baseAccuracy = baseHistory.history.acc[0];
    const customHistory = await customModel.fit(
        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});
    const customLoss = customHistory.history.loss[0];
    const customAccuracy = customHistory.history.acc[0];

    // Plot loss / accuracy.
    lossValues.push({'batch': i, 'loss': baseLoss, 'set': 'baseTrain'});
    lossValues.push({'batch': i, 'loss': customLoss, 'set': 'customTrain'});
    ui.plotLosses(lossValues);

    if (testBatch != null) {
      accuracyValues.push(
          {'batch': i, 'accuracy': baseAccuracy, 'set': 'baseAccuracy'});
      accuracyValues.push(
          {'batch': i, 'accuracy': customAccuracy, 'set': 'customAccuracy'});
      ui.plotAccuracies(accuracyValues);
    }

    batch.xs.dispose();
    batch.labels.dispose();
    if (testBatch != null) {
      testBatch.xs.dispose();
      testBatch.labels.dispose();
    }

    await tf.nextFrame();
  }
}

async function showPredictions() {
  const testExamples = 100;
  const batch = data.nextTestBatch(testExamples);

  tf.tidy(() => {
    const baseOutput = baseModel.predict(batch.xs.reshape([-1, 28, 28, 1]));
    const customOutput = customModel.predict(batch.xs.reshape([-1, 28, 28, 1]));

    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const basePredictions = Array.from(baseOutput.argMax(axis).dataSync());
    const customPredictions = Array.from(customOutput.argMax(axis).dataSync());

    ui.showTestResults(batch, basePredictions, customPredictions, labels);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  await load();
  await train();
  showPredictions();
}
mnist();
