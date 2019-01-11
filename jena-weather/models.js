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

import * as tf from '@tensorflow/tfjs';
import {JenaWeatherData} from './data';

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
 * Build a GRU model for the temperature-prediction problem.
 *
 * TODO(cais): Move this to a tfjs-node training script, as training
 *  the GRU in the browser turns out to be too slow.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @param {number} dropout Optional input dropout rate
 * @param {number} recurrentDropout Optional recurrent dropout rate.
 * @returns {tf.Model} A TensorFlow.js GRU model.
 */
function buildGRUModel(inputShape, dropout, recurrentDropout) {
  // TODO(cais): Add recurrent dropout.
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
 * @param {tf.Model} model A compiled tf.Model object.
 * @param {JenaWeatherData} jenaWeatherData A JenaWeatherData object.
 * @param {boolean} shuffle Whether the data is to be shuffled.
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
 * @param {number} epochEndCallback Optional callback to invoke at the
 *   end of every epoch.
 */
export async function trainModel(
    model, jenaWeatherData, normalize, includeDateTime, lookBack,
    step, delay, batchSize, epochs, displayEvery = 100,
    epochEndCallback) {
  const shuffle = true;
  const minIndex = 0;
  const maxIndex = 200000;
  const trainNextBatchFn = jenaWeatherData.getNextBatchFunction(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize,
      includeDateTime);

  const batchesPerEpoch = 500;
  for (let i = 0; i < epochs; ++i) {
    const t0 = tf.util.now();
    let totalTrainLoss = 0;
    let numSeen = 0;
    for (let j = 0; j < batchesPerEpoch; ++j) {
      const item = trainNextBatchFn();
      const trainLoss = await model.trainOnBatch(item.value[0], item.value[1]);

      numSeen += item.value[0].shape[0];
      totalTrainLoss += item.value[0].shape[0] * trainLoss;
      if ((j + 1) % displayEvery === 0) {
        const t = tf.util.now();
        const millisPerBatch = (t - t0) / (j + 1);
        console.log(
            `epoch ${i + 1}/${epochs} batch ${j + 1}/${batchesPerEpoch}: ` +
            `trainLoss=${trainLoss.toFixed(6)} ` +
            `(${millisPerBatch.toFixed(1)} ms/batch)`);
      }
      tf.dispose(item.value);
    }
    const t1 = tf.util.now();
    const epochTrainLoss = totalTrainLoss / numSeen;

    // Perform validation.
    const valMinIndex = 200001;
    const valMaxIndex = 300000;
    const valNextBatchFn = jenaWeatherData.getNextBatchFunction(
        false, lookBack, delay, batchSize, step, valMinIndex, valMaxIndex,
        normalize, includeDateTime);
    const valT0 = tf.util.now();
    const valSteps = Math.floor((300000 - 200001 - lookBack) / batchSize);
    tf.tidy(() => {
      console.log(`Running validation: valSteps=${valSteps}`);
      let totalValLoss = tf.scalar(0);
      numSeen = 0;
      for (let j = 0; j < valSteps; ++j) {
        if (j % displayEvery === 0) {
          console.log(`  Validation: step ${j}/${valSteps}`);
        }
        const item = valNextBatchFn();
        const evalOut =
            model.evaluate(item.value[0], item.value[1], {batchSize});
        const numExamples = item.value[0].shape[0];
        totalValLoss = tf.tidy(
            () => totalValLoss.add(evalOut.mulStrict(tf.scalar(numExamples))));
        numSeen += numExamples;
        tf.dispose([item.value, evalOut]);
      }
      const valLoss = totalValLoss.divStrict(tf.scalar(numSeen)).dataSync()[0];
      const valT1 = tf.util.now();
      const valMsPerBatch = (valT1 - valT0) / valSteps;
      console.log(
          `epoch ${i + 1}/${epochs}: trainLoss=${epochTrainLoss.toFixed(6)}; ` +
          `valLoss=${valLoss.toFixed(6)} ` +
          `(train: ${((t1 - t0) / batchesPerEpoch).toFixed(1)} ms/batch; ` +
          `val: ${valMsPerBatch.toFixed(1)} ms/batch)\n`);

      if (epochEndCallback) {
        epochEndCallback(i + 1, epochTrainLoss, valLoss)
      }
      tf.dispose(valLoss);
    });
  }

  return model;
}
