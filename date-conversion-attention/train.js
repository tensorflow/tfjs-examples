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

const tf = require('@tensorflow/tfjs');
// TODO(cais): Put under a command-line arg.
require('@tensorflow/tfjs-node');

const dateFormat = require('./date_format');

class GetLastTimestepLayer extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.supportMasking = true;
  }

  computeOutputShape(inputShape) {
    const outputShape = inputShape.slice();
    outputShape.splice(outputShape.length - 2, 1);
    return outputShape;
  }

  call(input, kwargs) {
    if (Array.isArray(input)) {
      input = input[0];
    }
    // console.log('In GetLastTimestepLayer.call():', input);  // DEBUG
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `Invalid input rank: ${inputRank}`);
    return tf.squeeze(tf.gather(input, [input.shape[1] - 1], 1), [1]);
  }

  static get className() {
    return 'GetLastTimestepLayer';
  }
}

function createModel(inputDictSize, outputDictSize, inputLength, outputLength) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  const encoderInput = tf.input({shape: [inputLength]});
  const decoderInput = tf.input({shape: [outputLength]});

  let encoder = tf.layers.embedding({
    inputDim: inputDictSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  }).apply(encoderInput);
  encoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(encoder);

//   console.log(encoder);  // DEBUG
  const encoderLast = new GetLastTimestepLayer({
    name: 'encoderLast'
  }).apply(encoder);

  let decoder = tf.layers.embedding({
    inputDim: outputDictSize,
    outputDim: embeddingDims,
    inputLength: outputLength,
    maskZero: true
  }).apply(decoderInput);
  decoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(decoder, {initialState: [encoderLast, encoderLast]});

  let attention = tf.layers.dot({axes: [2, 2]}).apply([decoder, encoder]);
  attention = tf.layers.activation({
    activation: 'softmax',
    name: 'attention'
  }).apply(attention);

  const context = tf.layers.dot({
    axes: [2, 1],
    name: 'context'
  }).apply([attention, encoder]);
  const deocderCombinedContext =
      tf.layers.concatenate().apply([context, decoder]);
  let output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: lstmUnits,
      activation: 'tanh'
    })
  }).apply(deocderCombinedContext);
  output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: outputDictSize,
      activation: 'softmax'
    })
  }).apply(output);

  const model = tf.model({
    inputs: [encoderInput, decoderInput],
    outputs: output
    // outputs: attention
  });
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });
  return model;
}

// DEBUG
async function run() {
  const model = createModel(
      dateFormat.INPUT_VOCAB.length, dateFormat.OUTPUT_VOCAB.length,
      dateFormat.INPUT_LENGTH, dateFormat.OUTPUT_LENGTH);
  model.summary();

  const numExamples = 1024;
  const encoderInput = tf.ones([numExamples, dateFormat.INPUT_LENGTH]);
  const decoderInput = tf.ones([numExamples, dateFormat.OUTPUT_LENGTH]);
  const decoderOutput = tf.ones(
      [numExamples, dateFormat.OUTPUT_LENGTH, dateFormat.OUTPUT_VOCAB.length]);
  // const out = model.predict([x, y]);
  // out.print();
  // console.log(out.shape);
  const history = await model.fit(
      [encoderInput, decoderInput], decoderOutput, {epochs: 10});
  console.log(history.history);
}

run();
