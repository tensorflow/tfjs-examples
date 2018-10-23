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

function falsePositives(yTrue, yPred) {
  return tf.tidy(() => {
    const one = tf.scalar(1);
    const zero = tf.scalar(0);
    return tf.logicalAnd(yTrue.equal(zero), yPred.equal(one))
        .sum()
        .cast('float32');
  });
}

function trueNegatives(yTrue, yPred) {
  return tf.tidy(() => {
    const zero = tf.scalar(0);
    return tf.logicalAnd(yTrue.equal(zero), yPred.equal(zero))
        .sum()
        .cast('float32');
  });
}

// TODO(cais): Use tf.metrics.falsePositiveRate when available.
function falsePositiveRate(yTrue, yPred) {
  return tf.tidy(() => {
    const fp = falsePositives(yTrue, yPred);
    const tn = trueNegatives(yTrue, yPred);
    return fp.div(fp.add(tn));
  });
}

/**
 * Draw a ROC curve.
 * 
 * @param {tf.Tensor} targets The actual target labels, as a 1D Tensor
 *   object consisting of only 0 and 1 values.
 * @param {tf.Tensor} probs The probabilities output by a model, as a 1D
 *   Tensor of the same shape as `targets`. It is assumed that the values of
 *   the elements are >=0 and <= 1.
 * @param {number} epoch The epoch number where the `probs` values come
 *   from.
 * @returns {number} Area under the curve (AUC).
 */
function drawROC(targets, probs, epoch) {
  return tf.tidy(() => {
    const thresholds = [
      0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
      0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
      0.9, 0.92, 0.94, 0.96, 0.98, 1.0
    ];
    const tprs = [];  // True positive rates.
    const fprs = [];  // False positive rates.
    let area = 0;
    for (let i = 0; i < thresholds.length; ++i) {
      const threshold = thresholds[i];

      const threshPredictions = utils.binarize(probs, threshold).as1D();
      const fpr = falsePositiveRate(targets, threshPredictions).get();
      const tpr = tf.metrics.recall(targets, threshPredictions).get();
      fprs.push(fpr);
      tprs.push(tpr);

      // Accumulate to area for AUC calculation.
      if (i > 0) {
        area += (tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i]) / 2;
      }
    }
    ui.plotROC(fprs, tprs, epoch);
    return area;
  });
}

// Some hyperparameters for model training.
const epochs = 400;
const batchSize = 350;

const data = new WebsitePhishingDataset();
data.loadData().then(async () => {
  await ui.updateStatus('Getting training and testing data...');
  const trainData = data.getTrainData();
  const testData = data.getTestData();

  await ui.updateStatus('Building model...');
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {inputShape: [data.numFeatures], units: 100, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 100, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  model.compile(
      {optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});

  let trainLoss;
  let valLoss;
  let trainAcc;
  let valAcc;
  let auc;

  await ui.updateStatus('Training starting...');
  await model.fit(trainData.data, trainData.target, {
    batchSize,
    epochs,
    validationSplit: 0.2,
    callbacks: {
    onEpochBegin: async (epoch) => {
        // Draw ROC every a few epochs.
        if ((epoch + 1)% 100 === 0 ||
            epoch === 0 || epoch === 2 || epoch === 4) {
            const probs = model.predict(testData.data);
            auc = drawROC(testData.target, probs, epoch);
        }
      },
      onEpochEnd: async (epoch, logs) => {
        await ui.updateStatus(`Epoch ${epoch + 1} of ${epochs} completed.`);

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
  tf.tidy(() => {
    const result =
        model.evaluate(testData.data, testData.target, {batchSize: batchSize});

    const testLoss = result[0].get();
    const testAcc = result[1].get();

    const probs = model.predict(testData.data);
    const predictions = utils.binarize(probs).as1D();

    const precision = tf.metrics.precision(testData.target, predictions).get();
    const recall = tf.metrics.recall(testData.target, predictions).get();
    const fpr = falsePositiveRate(testData.target, predictions).get();
    ui.updateStatus(
        `Final train-set loss: ${trainLoss.toFixed(4)} accuracy: ${
            trainAcc.toFixed(4)}\n` +
        `Final validation-set loss: ${valLoss.toFixed(4)} accuracy: ${
            valAcc.toFixed(4)}\n` +
        `Test-set loss: ${testLoss.toFixed(4)} accuracy: ${
            testAcc.toFixed(4)}\n` +
        `Precision: ${precision.toFixed(4)}\n` +
        `Recall: ${recall.toFixed(4)}\n` +
        `False positive rate (FPR): ${fpr.toFixed(4)}\n` + 
        `Area under the curve (AUC): ${auc.toFixed(4)}`);
  });
});
