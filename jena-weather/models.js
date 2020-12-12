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
 * Creating and training `tf.LayersModel`s for the temperature prediction
 * problem.
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
  const dataset = tf.data.generator(
      () => jenaWeatherData.getNextBatchFunction(
          false, lookBack, delay, batchSize, step, VAL_MIN_ROW, VAL_MAX_ROW,
          normalize, includeDateTime));

  const batchMeanAbsoluteErrors = [];
  const batchSizes = [];
  await dataset.forEachAsync(dataItem => {
    const features = dataItem.xs;
    const targets = dataItem.ys;
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
 * @returns {tf.LayersModel} A TensorFlow.js tf.LayersModel instance.
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
 * @returns {tf.LayersModel} A TensorFlow.js tf.LayersModel instance.
 */
export function buildMLPModel(inputShape, kernelRegularizer, dropoutRate) {
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
 * @returns {tf.LayersModel} A TensorFlow.js model consisting of a simpleRNN
 *   layer.
 */
export function buildSimpleRNNModel(inputShape) {
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.simpleRNN({units: rnnUnits, inputShape}));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {number} dropout Optional input dropout rate
 * @param {number} recurrentDropout Optional recurrent dropout rate.
 * @returns {tf.LayersModel} A TensorFlow.js GRU model.
 */
export function buildGRUModel(inputShape, dropout, recurrentDropout) {
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
 * @returns A compiled instance of `tf.LayersModel`.
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
 * @param {tf.LayersModel} model A compiled tf.LayersModel object. It is
 *   expected to have a 3D input shape `[numExamples, timeSteps, numFeatures].`
 *   and an output shape `[numExamples, 1]` for predicting the temperature
 * value.
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
 * @param {tf.Callback | tf.CustomCallbackArgs} customCallback Optional callback
 *   to invoke at the end of every epoch. Can optionally have `onBatchEnd` and
 *   `onEpochEnd` fields.
 */
export async function trainModel(
    model, jenaWeatherData, normalize, includeDateTime, lookBack, step, delay,
    batchSize, epochs, customCallback) {
  const trainShuffle = true;
  const trainDataset =
      tf.data
          .generator(
              () => jenaWeatherData.getNextBatchFunction(
                  trainShuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW,
                  TRAIN_MAX_ROW, normalize, includeDateTime))
          .prefetch(8);
  const evalShuffle = false;
  const valDataset = tf.data.generator(
      () => jenaWeatherData.getNextBatchFunction(
          evalShuffle, lookBack, delay, batchSize, step, VAL_MIN_ROW,
          VAL_MAX_ROW, normalize, includeDateTime));

  await model.fitDataset(trainDataset, {
    batchesPerEpoch: 500,
    epochs,
    callbacks: customCallback,
    validationData: valDataset
  });
}
