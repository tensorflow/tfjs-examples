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

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const timer = require('node-simple-timer');
const data = require('./data');

const NUM_EPOCHS = 700;
const BATCH_SIZE = 50;
const TEST_SIZE = 173;
const LEARNING_RATE = 0.01;

const weights = tf.variable(tf.randomNormal(shape = [data.num_features, 1]));
const bias = tf.variable(tf.randomNormal(shape = [1]));

const optimizer = tf.train.sgd(LEARNING_RATE);

const model = (input) => {
  return tf.tidy(() => {
    const output = input.matMul(weights).add(bias);
    return output;
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
  const totalTimer = new timer.Timer();
  totalTimer.start();

  await data.loadData();

  const epochTimer = new timer.Timer();
  for (let i = 0; i < NUM_EPOCHS; i++) {
    epochTimer.start();
    const trainSteps = await train();
    epochTimer.end();
    data.resetTraining();

    const time = epochTimer.seconds().toFixed(2);
    const stepsSec = (trainSteps / epochTimer.seconds()).toFixed(2);
    console.log(
        `* End Epoch: ${i + 1}: time: ${time}secs (${stepsSec} steps/sec)`);

    await test();
  }

  totalTimer.end();
  const time = totalTimer.seconds().toFixed(2);
  console.log(`**** Trained ${NUM_EPOCHS} epochs in ${time} secs`);
}

run();
