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

import {baseMnistModel} from './base_mnist_model';
import {customMnistModel} from './custom_mnist_model';
import {MnistData} from './data';
import * as ui from './ui';

// Create a convolutional MNIST model, as outlined in
// @tensorflow/tfjs-examples/mnist
const baseModel = baseMnistModel();
// Create an augmented MNIST model with a custom layer.  See inside
// custom_mnist_model.js to see how it is done.
const customModel = customMnistModel();


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
