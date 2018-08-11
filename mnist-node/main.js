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
require('@tensorflow/tfjs-node-gpu');

const timer = require('node-simple-timer');
const ProgressBar = require('progress');
const data = require('./data');
const model = require('./model');

const NUM_EPOCHS = 10;
const BATCH_SIZE = 100;
const TEST_SIZE = 50;

async function test() {
  if (!data.hasMoreTestData()) {
    data.resetTest();
  }
  const evalData = data.nextTestBatch(TEST_SIZE);
  const output = model.predict(evalData.image);
  const predictions = output.argMax(1).dataSync();
  const labels = evalData.label.argMax(1).dataSync();

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

  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  model.summary();

  let progressBar;
  let epochBeginTime;
  let millisPerStep;
  const epochs = 20;
  const batchSize = 128;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch =
      trainImages.shape[0] * (1 - validationSplit);
  const numTrainBatchesPerEpoch =
      Math.ceil(numTrainExamplesPerEpoch / batchSize);
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
    callbacks: {
      onEpochBegin: async (epoch) => {
        progressBar = new ProgressBar(
            ':bar: :eta', {total: numTrainBatchesPerEpoch, head: `>`});
        console.log(`Epoch ${epoch + 1} / ${epochs}`);
        epochBeginTime = tf.util.now();
      },
      onBatchEnd: async (batch, logs) => {
        if (batch === numTrainBatchesPerEpoch - 1) {
          millisPerStep =
              (tf.util.now() - epochBeginTime) / numTrainExamplesPerEpoch;
        }
        progressBar.tick();
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(
            `Loss: ${logs.loss.toFixed(3)} (train), ` +
            `${logs.val_loss.toFixed(3)} (val); ` +
            `Accuracy: ${logs.acc.toFixed(3)} (train), ` +
            `${logs.val_acc.toFixed(3)} (val) ` +
            `(${millisPerStep.toFixed(2)} ms/step)`);
        await tf.nextFrame();
      }
    }
  });

  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log();
  console.log('Evaluation result:');
  console.log(
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Acurracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
}

run();
