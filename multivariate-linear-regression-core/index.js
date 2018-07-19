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

data.loadData().then(async () => {
  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const NUM_EPOCHS = 500;
  const BATCH_SIZE = 50;
  const LEARNING_RATE = 0.01;

  const sgd = tf.train.sgd(LEARNING_RATE);

  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [data.numFeatures], units: 1}));
  model.compile({optimizer: sgd, loss: 'meanSquaredError'});

  await model.fit(trainData.data, trainData.target, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`* Epoch ${epoch}`);
        console.log(`* Train set loss: ${logs.loss.toFixed(4)}`);
        console.log(`* Validation set loss: ${logs.val_loss.toFixed(4)}`);

        // tf.nextFrame makes program wait till requestAnimationFrame
        // has completed. This helps mitigate blocking of UI thread
        // and thus browser tab.
        await tf.nextFrame();
      }
    }
  });

  const result =
      model.evaluate(testData.data, testData.target, {batchSize: BATCH_SIZE});

  const loss = result.get().toFixed(4);

  console.log(`* Test set loss: ${loss}`);
});
