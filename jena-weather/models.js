/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

/**
 * Creating and training `tf.Model`s for the temperature prediction problem.
 *
 * This file is used to create models for both
 * - the browser: see [index.js](./index.js), and
 * - the Node.js backend environment: see [train-rnn.js](./train-rnn.js).
 */

import * as tf from '@tensorflow/tfjs';
import {JenaWeatherData} from './data';

// Row ranges of the training and validation data subsets.
const TRAIN_MIN_ROW = 0;
const TRAIN_MAX_ROW = 200000;
const VAL_MIN_ROW = 200001;
const VAL_MAX_ROW = 300000;

/**
 * Calculate the commonsense baseline temperture-prediction accuracy.
 *
 * The latest value in the temperature feature column is used as the
 * prediction.
 *
 * @param {boolean} normalize Whether to used normalized data for training.
 * @param {boolean} includeDateTime Whether to include date and time features
 *   in training.
 * @param {number} lookBack Number of look-back time steps.
 * @param {number} step Step size used to generate the input features.
 * @param {number} delay How many steps in the future to make the prediction
 *   for.
 * @returns {number} The mean absolute error of the commonsense baseline
 *   prediction.
 */
export async function getBaselineMeanAbsoluteError(
    jenaWeatherData, normalize, includeDateTime, lookBack, step, delay) {
  const batchSize = 128;
  const nextBatchFn = jenaWeatherData.getNextBatchFunction(
      false, lookBack, delay, batchSize, step, VAL_MIN_ROW, VAL_MAX_ROW,
      normalize, includeDateTime);
  const dataset = tf.data.generator(nextBatchFn);

  const batchMeanAbsoluteErrors = [];
  const batchSizes = [];
  await dataset.forEach(dataItem => {
    const features = dataItem[0];
    const targets = dataItem[1];
    const timeSteps = features.shape[1];
    batchSizes.push(features.shape[0]);
    batchMeanAbsoluteErrors.push(tf.tidy(
        () => tf.losses.absoluteDifference(
            targets,
            features.gather([timeSteps - 1], 1).gather([1], 2).squeeze([2]))));
  });
  const meanAbsoluteError = tf.tidy(() => {
    const batchSizesTensor = tf.tensor1d(batchSizes);
    const batchMeanAbsoluteErrorsTensor = tf.stack(batchMeanAbsoluteErrors);
    return batchMeanAbsoluteErrorsTensor.mul(batchSizesTensor)
        .sum()
        .div(batchSizesTensor.sum());
  });
  tf.dispose(batchMeanAbsoluteErrors);
  return meanAbsoluteError.dataSync()[0];
}

/**
 * Build a linear-regression model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.Model} A TensorFlow.js tf.Model instance.
 */
function buildLinearRegressionModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape}));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {tf.regularizer.Regularizer} kernelRegularizer An optional
 *   regularizer for the kernel of the first (hdiden) dense layer of the MLP.
 *   If not specified, no weight regularization will be included in the MLP.
 * @param {number} dropoutRate Dropout rate of an optional dropout layer
 *   inserted between the two dense layers of the MLP. Optional. If not
 *   specified, no dropout layers will be included in the MLP.
 * @returns {tf.Model} A TensorFlow.js tf.Model instance.
 */
function buildMLPModel(inputShape, kernelRegularizer, dropoutRate) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape}));
  model.add(
      tf.layers.dense({units: 32, kernelRegularizer, activation: 'relu'}));
  if (dropoutRate > 0) {
    model.add(tf.layers.dropout({rate: dropoutRate}));
  }
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a simpleRNN-based model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.Model} A TensorFlow.js model consisting of a simpleRNN layer.
 */
function buildSimpleRNNModel(inputShape) {
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.simpleRNN({
    units: rnnUnits,
    inputShape
  }));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {number} dropout Optional input dropout rate
 * @param {number} recurrentDropout Optional recurrent dropout rate.
 * @returns {tf.Model} A TensorFlow.js GRU model.
 */
function buildGRUModel(inputShape, dropout, recurrentDropout) {
  // TODO(cais): Recurrent dropout is currently not fully working.
  //   Make it work and add a flag to train-rnn.js.
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.gru({
    units: rnnUnits,
    inputShape,
    dropout: dropout || 0,
    recurrentDropout: recurrentDropout || 0
  }));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a model for the temperature-prediction problem.
 *
 * @param {string} modelType Model type.
 * @param {number} numTimeSteps Number of time steps in each input.
 *   exapmle
 * @param {number} numFeatures Number of features (for each time step).
 * @returns A compiled instance of `tf.Model`.
 */
export function buildModel(modelType, numTimeSteps, numFeatures) {
  const inputShape = [numTimeSteps, numFeatures];

  console.log(`modelType = ${modelType}`);
  let model;
  if (modelType === 'mlp') {
    model = buildMLPModel(inputShape);
  } else if (modelType === 'mlp-l2') {
    model = buildMLPModel(inputShape, tf.regularizers.l2());
  } else if (modelType === 'linear-regression') {
    model = buildLinearRegressionModel(inputShape);
  } else if (modelType === 'mlp-dropout') {
    const regularizer = null;
    const dropoutRate = 0.25;
    model = buildMLPModel(inputShape, regularizer, dropoutRate);
  } else if (modelType === 'simpleRNN') {
    model = buildSimpleRNNModel(inputShape);
  } else if (modelType === 'gru') {
    model = buildGRUModel(inputShape);
    // TODO(cais): Add gru-dropout with recurrentDropout.
  } else {
    throw new Error(`Unsupported model type: ${modelType}`);
  }

  model.compile({loss: 'meanAbsoluteError', optimizer: 'rmsprop'});
  model.summary();
  return model;
}

/**
 * Train a model on the Jena weather data.
 *
 * @param {tf.Model} model A compiled tf.Model object. It is expected to
 *   have a 3D input shape `[numExamples, timeSteps, numFeatures].` and an
 *   output shape `[numExamples, 1]` for predicting the temperature value.
 * @param {JenaWeatherData} jenaWeatherData A JenaWeatherData object.
 * @param {boolean} normalize Whether to used normalized data for training.
 * @param {boolean} includeDateTime Whether to include date and time features
 *   in training.
 * @param {number} lookBack Number of look-back time steps.
 * @param {number} step Step size used to generate the input features.
 * @param {number} delay How many steps in the future to make the prediction
 *   for.
 * @param {number} batchSize batchSize for training.
 * @param {number} epochs Number of training epochs.
 * @param {number} displayEvery Log info to console every _ batches.
 * @param {number} customCallbacks Optional callback args to invoke at the
 *   end of every epoch. Can optionally have `onBatchEnd` and `onEpochEnd`
 *   fields.
 */
export async function trainModel(
    model, jenaWeatherData, normalize, includeDateTime, lookBack, step, delay,
    batchSize, epochs, displayEvery = 100, customCallbacks) {
  const shuffle = true;

  const trainNextBatchFn = jenaWeatherData.getNextBatchFunction(
      shuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW, TRAIN_MAX_ROW,
      normalize, includeDateTime);
  const trainDataset = tf.data.generator(trainNextBatchFn).prefetch(8);

  const batchesPerEpoch = 500;
  let t0;
  let currentEpoch;
  await model.fitDataset(trainDataset, {
    batchesPerEpoch,
    epochs,
    callbacks: {
      onEpochBegin: async (epoch) => {
        currentEpoch = epoch;
        t0 = tf.util.now();
      },
      onBatchEnd: async (batch, logs) => {
        if ((batch + 1) % displayEvery === 0) {
          const t = tf.util.now();
          const millisPerBatch = (t - t0) / (batch + 1);
          console.log(
              `epoch ${currentEpoch + 1}/${epochs} ` +
              `batch ${batch + 1}/${batchesPerEpoch}: ` +
              `loss=${logs.loss.toFixed(6)} ` +
              `(${millisPerBatch.toFixed(1)} ms/batch)`);
          if (customCallbacks && customCallbacks.onBatchEnd) {
            customCallbacks.onBatchEnd(batch, logs);
          }
        }
      },
      onEpochEnd: async (epoch, logs) => {
        const valNextBatchFn = jenaWeatherData.getNextBatchFunction(
            false, lookBack, delay, batchSize, step, VAL_MIN_ROW, VAL_MAX_ROW,
            normalize, includeDateTime);
        const valDataset = tf.data.generator(valNextBatchFn);
        console.log(`epoch ${epoch + 1}/${epochs}: Performing validation...`);
        // TODO(cais): Remove the second arg (empty object), when the bug is
        // fixed:
        //   https://github.com/tensorflow/tfjs/issues/1096
        const evalOut = await model.evaluateDataset(valDataset, {});
        logs.val_loss = (await evalOut.data())[0];
        tf.dispose(evalOut);
        console.log(
            `epoch ${epoch + 1}/${epochs}: ` +
            `trainLoss=${logs.loss.toFixed(6)}; ` +
            `valLoss=${logs.val_loss.toFixed(6)}`);
        if (customCallbacks && customCallbacks.onEpochEnd) {
          customCallbacks.onEpochEnd(epoch, logs);
        }
      }
    }
  });

  return model;
}
