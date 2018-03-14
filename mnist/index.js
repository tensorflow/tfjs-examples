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

import {MnistData} from './data';
import * as ui from './ui';

const LEARNING_RATE = 0.1;
const BATCH_SIZE = 64;

const model = tf.sequential({
  layers: [
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }),
    tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}),
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }),
    tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}),
    tf.layers.flatten(), tf.layers.dense({
      units: 10,
      useBias: 10,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    })
  ]
});

// TODO(nsthorat): Use tf.train.sgd() once compile supports core optimizers.
const optimizer = new tf.train.sgd(LEARNING_RATE);
console.log('optimizer =', optimizer);  // DEBUG
model.compile({optimizer, loss: 'categoricalCrossentropy'});

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function train() {
  ui.isTraining();
  for (let i = 0; i < 100; i++) {
    const batch = data.nextTrainBatch(BATCH_SIZE);
    // The entire dataset doesn't fit into memory so we call fit repeatedly with
    // batches.
    const history = await model.fit({
      x: batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
      y: batch.labels,
      batchSize: BATCH_SIZE,
      epochs: 1
    });
    console.log('loss:', history.history.loss[0]);

    batch.xs.dispose();
    batch.labels.dispose();
  }
}

async function test() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

  const axis = 1;
  const labels = Array.from(batch.labels.argMax(axis).dataSync());
  const predictions = Array.from(output.argMax(axis).dataSync());

  ui.showTestResults(batch, predictions, labels);
}

async function mnist() {
  await load();
  await train();
  test();
}
mnist();
