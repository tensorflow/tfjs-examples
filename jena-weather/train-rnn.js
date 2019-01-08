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
// TODO(cais): Add --gpu flag.
import '@tensorflow/tfjs-node';
import {JenaWeatherData} from './data';

global.fetch = require('node-fetch');

/**
 * Build a GRU model for the temperature-prediction problem.
 *
 * TODO(cais): Move this to a tfjs-node training script, as training
 *  the GRU in the browser turns out to be too slow.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.Model} A TensorFlow.js GRU model.
 */
function buildGRUModel(inputShape) {
  // TODO(cais): Add recurrent dropout.
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.gru({units: rnnUnits, inputShape}));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

async function main() {
  // TODO(cais): Make it possible to load the data from a local file.
  const jenaWeatherData = new JenaWeatherData();
  console.log(`Loading Jena weather data...`);
  await jenaWeatherData.load();

  // TODO(cais): De-duplicate code.
  const shuffle = true;
  const lookBack = 10 * 24 * 6;  // Look back 10 days.
  const step = 6;                // 1-hour steps.
  const delay = 24 * 6;          // Predict the weather 1 day later.
  const batchSize = 128;
  const minIndex = 0;
  const maxIndex = 200000;
  const normalize = true;
  const includeDateTime = false;
  const numFeatures = jenaWeatherData.getDataColumnNames().length;
  const inputShape = [Math.floor(lookBack / step), numFeatures];
  const model = buildGRUModel(inputShape);
  model.compile({loss: 'meanAbsoluteError', optimizer: 'rmsprop'});
  model.summary();  // DEBUG

  const trainNextBatchFn = jenaWeatherData.getNextBatchFunction(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize,
      includeDateTime);

  // TODO(cais): De-duplicate code.
  const epochs = 20;  // TODO(cais): Do not hardcode.
  const batchesPerEpoch = 500;
  const displayEvery = 1;
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
        console.log(
            `epoch ${i + 1}/${epochs} batch ${j + 1}/${batchesPerEpoch}: ` +
            `trainLoss=${trainLoss.toFixed(6)}`);
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
      plotLoss(modelType, i + 1, epochTrainLoss, valLoss)

      tf.dispose(valLoss);
    });
  }
}

main();
