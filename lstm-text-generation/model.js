/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export function createModel(sampleLen, charSetSize, lstmLayerSizes) {
  if (!Array.isArray(lstmLayerSizes)) {
    lstmLayerSizes = [lstmLayerSizes];
  }

  const model = tf.sequential();
  for (let i = 0; i < lstmLayerSizes.length; ++i) {
    const lstmLayerSize = lstmLayerSizes[i];
    model.add(tf.layers.lstm({
      units: lstmLayerSize,
      returnSequences: i < lstmLayerSizes.length - 1,
      inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
    }));
  }
  model.add(
      tf.layers.dense({units: charSetSize, activation: 'softmax'}));

  return model;
}

export function compileModel(model, learningRate) {
  const optimizer = tf.train.rmsprop(learningRate);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  console.log(`Compiled model with learning rate ${learningRate}`);
  model.summary();
}

export async function fitModel(
    model, textData, numEpochs, examplesPerEpoch, batchSize, validationSplit,
    callbacks) {
  for (let i = 0; i < numEpochs; ++i) {
    const [xs, ys] = textData.nextDataEpoch(examplesPerEpoch);
    await model.fit(xs, ys, {
      epochs: 1,
      batchSize: batchSize,
      validationSplit,
      callbacks
    });
    xs.dispose();
    ys.dispose();
  }
}
