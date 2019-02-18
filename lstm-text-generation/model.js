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

/**
 * Generate text using a next-char-prediction model.
 *
 * @param {tf.Model} model The model object to be used for the text generation,
 *   assumed to have input shape `[null, sampleLen, charSetSize]` and output
 *   shape `[null, charSetSize]`.
 * @param {number[]} sentenceIndices The character indices in the seed sentence.
 * @param {number} length Length of the sentence to generate.
 * @param {number} temperature Temperature value. Must be a number >= 0 and
 *   <= 1.
 * @param {(char: string) => Promise<void>} onTextGenerationChar An optinoal
 *   callback to be invoked each time a character is generated.
 * @returns {string} The generated sentence.
 */
export async function generateText(
    model, textData, sentenceIndices, length, temperature,
    onTextGenerationChar) {
  const sampleLen = model.inputs[0].shape[1];
  const charSetSize = model.inputs[0].shape[2];
  console.log(`generateText(): sampleLen=${sampleLen}, ` +
      `charSetSize=${charSetSize}, length=${length}`);  // DEBUG

  // Avoid overwriting the original input.
  sentenceIndices = sentenceIndices.slice();

  let generated = '';
  while (generated.length < length) {
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer =
        new tf.TensorBuffer([1, sampleLen, charSetSize]);
    for (let i = 0; i < sampleLen; ++i) {
      inputBuffer.set(1, 0, i, sentenceIndices[i]);
    }
    const input = inputBuffer.toTensor();

    // Call model.predict() to get the probability values of the next
    // character.
    const output = model.predict(input);

    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), temperature);
    const winnerChar = textData.getFromCharSet(winnerIndex);
    if (onTextGenerationChar != null) {
      await onTextGenerationChar(winnerChar);
    }

    generated += winnerChar;
    sentenceIndices = sentenceIndices.slice(1);
    sentenceIndices.push(winnerIndex);

    input.dispose();
    output.dispose();
  }
  return generated;
}

/**
 * Draw a sample based onprobabilities.
 *
 * @param {tf.Tensor} preds Predicted probabilities, as a 1D `tf.Tensor` of
 *   shape `[charSetSize]`.
 * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
 *   or diversity) to use during sampling. Number be a number > 0, as a Scalar
 *   `tf.Tensor`.
 * @returns {number} The 0-based index for the randomly-drawn sample, in the
 *   range of `[0, charSetSize - 1]`.
 */
function sample(preds, temperature) {
  return tf.tidy(() => {
    let isNormalized = false;
    if (tf.ENV.get('IS_BROWSER')) {
      const logPreds = tf.div(tf.log(preds), temperature);
      const expPreds = tf.exp(logPreds);
      const sumExpPreds = tf.sum(expPreds);
      preds = tf.div(expPreds, sumExpPreds);
      isNormalized = true;
    }
    // Treat preds a the probabilites of a multinomial distribution and
    // randomly draw a sample from the distribution.
    return tf.multinomial(preds, 1, null, isNormalized).dataSync()[0];
  });
}
