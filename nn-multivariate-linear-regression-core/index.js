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

const data = new BostonHousingDataset();

const NUM_EPOCHS = 700;
const BATCH_SIZE = 50;
const TEST_SIZE = 173;
const HIDDEN_UNITS = 10;

const weights1 = tf.variable(
    initializeWeights([data.num_features, HIDDEN_UNITS], data.num_features));
const bias1 = tf.variable(initializeWeights([HIDDEN_UNITS], data.num_features));

const weights2 =
    tf.variable(initializeWeights([HIDDEN_UNITS, HIDDEN_UNITS], HIDDEN_UNITS));
const bias2 = tf.variable(initializeWeights([HIDDEN_UNITS], HIDDEN_UNITS));

const weights3 =
    tf.variable(initializeWeights([HIDDEN_UNITS, 1], HIDDEN_UNITS));
const bias3 = tf.variable(initializeWeights([1], HIDDEN_UNITS));

const optimizer = tf.train.adam();

function initializeWeights(shape, prevLayerSize) {
  return tf.randomNormal(shape).mul(tf.scalar(Math.sqrt(2.0 / prevLayerSize)));
}

const model = (input) => {
  return tf.tidy(() => {
    const hiddenLayer1 = input.matMul(weights1).add(bias1).relu();
    const hiddenLayer2 = hiddenLayer1.matMul(weights2).add(bias2).relu();
    return hiddenLayer2.matMul(weights3).add(bias3);
  });
};

const calculateLoss = (targets, prediction) => {
  return tf.tidy(() => {
    return tf.losses.meanSquaredError(targets, prediction);
  });
};

async function train() {
  let step = 0;
  let returnLoss = true;

  while (data.hasMoreTrainingData()) {
    const batch = data.nextTrainBatch(BATCH_SIZE);
    const loss = optimizer.minimize(() => {
      const cost = calculateLoss(batch.target, model(batch.data));
      return cost;
    }, returnLoss);
    if (step && step % 2 === 0) {
      console.log(`  - step: ${step}: loss: ${loss.dataSync()}`);
    }
    await tf.nextFrame();
    step++;
  }

  return step;
}

async function test() {
  if (!data.hasMoreTestData()) {
    data.resetTest();
  }
  const evalData = data.nextTestBatch(TEST_SIZE);
  const predictions = model(evalData.data);
  const targets = evalData.target;

  const loss = calculateLoss(targets, predictions).dataSync();
  console.log(`* Test set loss: ${loss}\n`);
}

async function run() {
  const totalTimerStart = performance.now();

  await data.loadData();

  for (let i = 0; i < NUM_EPOCHS; i++) {
    const epochTimerStart = performance.now();
    const trainSteps = await train();
    const epochTimerEnd = performance.now();
    data.resetTraining();

    const time = ((epochTimerEnd - epochTimerStart) / 1000.0).toFixed(2);
    const stepsSec = (trainSteps / time).toFixed(2);
    console.log(
        `* End Epoch: ${i + 1}: time: ${time}secs (${stepsSec} steps/sec)`);

    await test();
  }

  const totalTimerEnd = performance.now();
  const time = ((totalTimerEnd - totalTimerStart) / 1000.0).toFixed(2);
  console.log(`**** Trained ${NUM_EPOCHS} epochs in ${time} secs`);
}

run();
