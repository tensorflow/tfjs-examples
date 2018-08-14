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

import {BostonHousingDataset} from './data';
import * as normalization from './normalization';
import * as ui from './ui';

// Some hyperparameters for model training.
const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const data = new BostonHousingDataset();

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export const linearRegressionModel = () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [data.numFeatures], units: 1}));

  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export const multiLayerPerceptronRegressionModel = () => {
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {inputShape: [data.numFeatures], units: 50, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1}));

  return model;
};

/**
 * Fetches training and testing data, normalizes, compiles `model`, trains the
 * model using train data and runs model against test data.
 *
 * @param {tf.Sequential} model Model to be trained.
 */
export const run = async (model) => {
  await ui.updateStatus('Getting training and testing data...');
  // Load data (possibly from remote CSV).
  const trainData = data.getTrainDataAsTensors();
  const testData = data.getTestDataAsTensors();
  // Normalize mean and standard deviation of data.
  let {means, stddevs} =
      normalization.determineMeanAndStddev(trainData.features);
  trainData.normalizedFeatures =
      normalization.normalizeTensor(trainData.features, means, stddevs);
  testData.normalizedFeatures =
      normalization.normalizeTensor(testData.features, means, stddevs);


  await ui.updateStatus('Compiling model...');

  model.compile(
      {optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError'});

  let trainLoss;
  let valLoss;
  await ui.updateStatus('Starting training process...');
  await model.fit(trainData.normalizedFeatures, trainData.target, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationData: [testData.normalizedFeatures, testData.target],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateStatus(`Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`);
        trainLoss = logs.loss;
        valLoss = logs.val_loss;
        await ui.plotData(epoch, trainLoss, valLoss);
        // tf.nextFrame makes the program wait until requestAnimationFrame()
        // has completed. This helps mitigate blocking of UI thread
        // and thus browser tab.
        await tf.nextFrame();
      }
    }
  });

  await ui.updateStatus('Running on test data...');
  const result = model.evaluate(
      testData.normalizedFeatures, testData.target, {batchSize: BATCH_SIZE});
  const testLoss = result.get();
  await ui.updateStatus(
      `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
      `Test-set loss: ${testLoss.toFixed(4)}`);
};

document.addEventListener('DOMContentLoaded', async () => {
  await data.loadData();
  await ui.updateStatus('Data loaded!');
  await ui.setup();
}, false);
