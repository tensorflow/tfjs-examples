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

import {WebsitePhishingDataset} from './data';
import * as ui from './ui';
import * as utils from './utils';

// Some hyperparameters for model training.
const NUM_EPOCHS = 100;  // TODO(cais): Change it back to 400 or 300. DO NOT SUBMIT.
const BATCH_SIZE = 350;

const data = new WebsitePhishingDataset();
data.loadData().then(async () => {
  await ui.updateStatus('Getting training and testing data...');
  const trainData = data.getTrainData();
  const testData = data.getTestData();

  await ui.updateStatus('Building model...');
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [data.numFeatures],
    units: 100,
    activation: 'sigmoid'
  }));
  model.add(tf.layers.dense({units: 100, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  let trainLoss;
  let valLoss;
  let trainAcc;
  let valAcc;

  await ui.updateStatus('Training starting...');
  await model.fit(trainData.data, trainData.target, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateStatus(`Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`);

        trainLoss = logs.loss;
        valLoss = logs.val_loss;
        trainAcc = logs.acc;
        valAcc = logs.val_acc;

        await ui.plotData(epoch, trainLoss, valLoss);
        await ui.plotAccuracies(epoch, trainAcc, valAcc);
      }
    }
  });

  await ui.updateStatus('Running on test data...');
  const result =
      model.evaluate(testData.data, testData.target, {batchSize: BATCH_SIZE});

  const testLoss = result[0].get();
  const testAcc = result[1].get();

  const predictions = utils.binarize(
      model.predict(testData.data, {batchSize: BATCH_SIZE}).as1D());

  const precision = tf.metrics.precision(testData.target, predictions).get();
  const recall = tf.metrics.recall(testData.target, predictions).get();

  await ui.updateStatus(
      `Final train-set loss: ${trainLoss.toFixed(4)} accuracy: ${
          trainAcc.toFixed(4)}\n` +
      `Final validation-set loss: ${valLoss.toFixed(4)} accuracy: ${
          valAcc.toFixed(4)}\n` +
      `Test-set loss: ${testLoss.toFixed(4)} accuracy: ${
          testAcc.toFixed(4)}\n` +
      `Precision: ${precision.toFixed(4)}\n` +
      `Recall: ${recall.toFixed(4)}`);
});
