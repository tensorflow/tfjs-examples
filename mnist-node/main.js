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
const ProgressBar = require('progress');

const data = require('./data');
const model = require('./model');

async function run() {
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

  console.log('\nEvaluation result:');
  console.log(
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Acurracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
}

run();
