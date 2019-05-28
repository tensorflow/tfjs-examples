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
import * as tfvis from '@tensorflow/tfjs-vis';

import {BostonHousingDataset, featureDescriptions} from './data';
// TODO(kangyi, soergel): Remove this once we have a public statistics API.
import {computeDatasetStatistics} from './stats';
import * as ui from './ui';

// Some hyperparameters for model training.
const NUM_EPOCHS = 200;

const BATCH_SIZE = 40;

const LEARNING_RATE = 0.01;

const preparedData = {
  trainData: null,
  validationData: null,
  testData: null
};

let bostonData;

// Convert loaded data into tensors and creates normalized versions of the
// features.
export async function loadDataAndNormalize() {
  // TODO(kangyizhang): Statistics should be generated from trainDataset
  // directly. Update following code after
  // https://github.com/tensorflow/tfjs-data/issues/32 is resolved.

  // Gets mean and standard deviation of data.
  // row[0] is feature data.
  const featureStats = await computeDatasetStatistics(
      bostonData.trainDataset.map((row) => row.xs));

  // Normalizes data.
  preparedData.trainData =
      bostonData.trainDataset.map(row => normalizeFeatures(row, featureStats))
          .batch(BATCH_SIZE);
  preparedData.validationData =
      bostonData.validationDataset
          .map(row => normalizeFeatures(row, featureStats))
          .batch(BATCH_SIZE);
  preparedData.testData =
      bostonData.testDataset.map(row => normalizeFeatures(row, featureStats))
          .batch(BATCH_SIZE);
}

/**
 * Normalizes features with statistics and returns a new object.
 */
// TODO(kangyizhang, bileschi): Replace these with preprocessing layers once
// they are available.
function normalizeFeatures(row, featureStats) {
  const features = row.xs;
  const normalizedFeatures = [];
  features.forEach(
      (value, index) => normalizedFeatures.push(
          (value - featureStats[index].mean) / featureStats[index].stddev));
  return {xs: normalizedFeatures, ys: row.ys};
}

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export function linearRegressionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [bostonData.numFeatures], units: 1}));

  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({units: 1}));

  model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
export function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense(
      {units: 50, activation: 'sigmoid', kernelInitializer: 'leCunNormal'}));
  model.add(tf.layers.dense({units: 1}));

  model.summary();
  return model;
};


/**
 * Describe the current linear weights for a human to read.
 *
 * @param {Array} kernel Array of floats of length 12.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value
 *     feature weight.
 */
export function describeKernelElements(kernel) {
  tf.util.assert(
      kernel.length == 12,
      `kernel must be a array of length 12, got ${kernel.length}`);
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({description: featureDescriptions[idx], value: kernel[idx]});
  }
  return outList;
}

/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 * @param {boolean} weightsIllustration Whether to print info about the learned
 *  weights.
 */
export async function run(model, modelName, weightsIllustration) {
  model.compile(
      {optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError'});

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus('Starting training process...');

  // Fit the model using the prepared Dataset.
  await model.fitDataset(preparedData.trainData, {
    epochs: NUM_EPOCHS,
    validationData: preparedData.validationData,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(
            `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`, modelName);
        // TODO(kangyizhang): Update this once
        // https://github.com/tensorflow/tfjs/issues/984 is fixed.
        trainLogs.push({loss: logs.loss, val_loss: logs.val_loss});
        tfvis.show.history(container, trainLogs, ['loss', 'val_loss']);

        if (weightsIllustration) {
          model.layers[0].getWeights()[0].data().then(kernelAsArr => {
            const weightsList = describeKernelElements(kernelAsArr);
            ui.updateWeightDescription(weightsList);
          });
        }
      }
    }
  });

  ui.updateStatus('Running on test data...');
  const result = (await model.evaluateDataset(preparedData.testData, {}));
  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  await ui.updateModelStatus(
      `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
          `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
          `Test-set loss: ${testLoss.toFixed(4)}`,
      modelName);
};

export const computeBaseline = async () => {
  // TODO(kangyizhang): Remove this once statistics support nested object.
  // row[1] is target data.
  const targetStats = await computeDatasetStatistics(
      bostonData.trainDataset.map((row) => row.ys));
  const avgPrice = targetStats[0].mean;
  console.log(`Average price: ${avgPrice}`);

  let testSquareError = 0;
  let testCount = 0;

  await bostonData.testDataset.forEach((row) => {
    testSquareError += Math.pow(row.ys - avgPrice, 2);
    testCount++;
  });

  if (testCount === 0) {
    throw new Error('No test data found!');
  }
  const baseline = testSquareError / testCount;
  console.log(`Baseline loss: ${baseline}`);
  const baselineMsg =
      `Baseline loss (meanSquaredError) is ${baseline.toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
};

document.addEventListener('DOMContentLoaded', async () => {
  bostonData = await BostonHousingDataset.create();
  ui.updateStatus('Data loaded, converting to tensors');
  await loadDataAndNormalize();
  ui.updateStatus(
      'Data is now available as tensors.\n' +
      'Click a train button to begin.');
  // TODO Explain what baseline loss is. How it is being computed in this
  // Instance
  ui.updateBaselineStatus('Estimating baseline loss');
  computeBaseline();
  await ui.setup();
}, false);
