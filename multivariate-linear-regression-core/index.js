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

const NUM_EPOCHS = 250;
const BATCH_SIZE = 50;
const LEARNING_RATE = 0.01;

const sgd = tf.train.sgd(LEARNING_RATE);

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [data.num_features], units: 1}));
model.compile({optimizer: sgd, loss: 'meanSquaredError'});

const train = async (epoch, trainData) => {
  await model.fit(trainData.data, trainData.target, {
    batchSize: BATCH_SIZE,
    epochs: 1,
    callbacks: {
      onEpochEnd: async (_, logs) => {
        console.log(`* Train set loss: ${logs.loss.toFixed(4)}`);

        // tf.nextFrame makes program wait till requestAnimationFrame
        // has completed. This helps mitigate blocking of UI thread
        // and thus browser tab.
        await tf.nextFrame();
      }
    }
  });
};

const run = async () => {
  const totalTimerStart = performance.now();

  await data.loadData();

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  for (let epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
    console.log(`** Start Epoch ${epoch} **`);

    const epochTimerStart = performance.now();
    await train(epoch, trainData);
    const epochTimerEnd = performance.now();

    const time = ((epochTimerEnd - epochTimerStart) / 1000.0).toFixed(2);

    const result =
        model.evaluate(testData.data, testData.target, {batchSize: BATCH_SIZE});

    const loss = result.get().toFixed(4);

    console.log(`* Test set loss: ${loss}`);
    console.log(`** End Epoch ${epoch}: time: ${time}secs **`);
  }

  const totalTimerEnd = performance.now();
  const time = ((totalTimerEnd - totalTimerStart) / 1000.0).toFixed(2);
  console.log(`**** Trained ${NUM_EPOCHS} epochs in ${time} secs ****`);
};

run();
