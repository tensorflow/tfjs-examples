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

const NUM_EPOCHS = 500;
const BATCH_SIZE = 50;
const TEST_SIZE = 173;
const LEARNING_RATE = 0.01;

const sgd = tf.train.sgd(LEARNING_RATE);

const model = tf.sequential();
model.add(tf.layers.dense({
  inputShape: [data.num_features],
  units: 1,
  kernelInitializer: 'randomNormal',
  biasInitializer: 'randomNormal',
  useBias: true
}));
model.compile({optimizer: sgd, loss: 'meanSquaredError'});

const train = async () => {
  let step = 0;

  while (data.hasMoreTrainingData()) {
    const batch = data.nextTrainBatch(BATCH_SIZE);
    const history = await model.fit(
        batch.data, batch.target, {batchSize: BATCH_SIZE, shuffle: true});

    if (step && step % 2 === 0) {
      const loss = history.history.loss[0].toFixed(6);
      console.log(`  - step: ${step}: loss: ${loss}`);
    }

    await tf.nextFrame();
    step++;
  }

  return step;
};

const test = async () => {
  if (!data.hasMoreTestData()) {
    data.resetTest();
  }

  const evalData = data.nextTestBatch(TEST_SIZE);
  const predictions = model.predict(evalData.data);
  const targets = evalData.target;

  const loss = tf.losses.meanSquaredError(targets, predictions).dataSync();
  console.log(`* Test set loss: ${loss}\n`);
};

const run = async () => {
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
};

run();
