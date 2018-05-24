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
tf.setBackend('tensorflow');

const timer = require('node-simple-timer');
const data = require('./data');
const model = require('./model');

const NUM_EPOCHS = 10;
const BATCH_SIZE = 100;
const TEST_SIZE = 50;

async function train() {
  let step = 0;
  while (data.hasMoreTrainingData()) {
    const batch = data.nextTrainBatch(BATCH_SIZE);
    const history = await model.fit(
        batch.image, batch.label, {batchSize: BATCH_SIZE, shuffle: false});

    if (step % 20 === 0) {
      console.log(`  - step: ${step}: loss: ${
          history.history.loss[0].toFixed(
              6)}, accuracy: ${history.history.acc[0].toFixed(4)}`);
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
  const output = model.predict(evalData.image);
  const predictions = Array.from(output.argMax(1).dataSync());
  const labels = Array.from(evalData.label.argMax(1).dataSync());

  let correct = 0;
  for (let i = 0; i < TEST_SIZE; i++) {
    if (predictions[i] === labels[i]) {
      correct++;
    }
  }
  const accuracy = ((correct / TEST_SIZE) * 100).toFixed(2);
  console.log(`* Test set accuracy: ${accuracy}%\n`);
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

    console.log(
        `* End Epoch: ${i + 1}: time: ${epochTimer.seconds().toFixed(2)}secs (${
            (trainSteps / epochTimer.seconds()).toFixed(2)} steps/sec)`);

    test();
  }

  totalTimer.end();
  console.log(`**** Trained ${NUM_EPOCHS} epochs in ${
      totalTimer.seconds().toFixed(2)} secs`);
}

run();
