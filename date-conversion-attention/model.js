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
import * as dateFormat from './date_format';

/**
 * A custom layer used to obtain the last time step of an RNN sequential
 * output.
 */
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

  call(input) {
    if (Array.isArray(input)) {
      input = input[0];
    }
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `Invalid input rank: ${inputRank}`);
    return input.gather([input.shape[1] - 1], 1).squeeze([1]);
  }

  static get className() {
    return 'GetLastTimestepLayer';
  }
}
tf.serialization.registerClass(GetLastTimestepLayer);

/**
 * Create an LSTM-based attention model for date conversion.
 *
 * @param {number} inputVocabSize Input vocabulary size. This includes
 *   the padding symbol. In the context of this model, "vocabulary" means
 *   the set of all unique characters that might appear in the input date
 *   string.
 * @param {number} outputVocabSize Output vocabulary size. This includes
 *   the padding and starting symbols. In the context of this model,
 *   "vocabulary" means the set of all unique characters that might appear in
 *   the output date string.
 * @param {number} inputLength Maximum input length (# of characters). Input
 *   sequences shorter than the length must be padded at the end.
 * @param {number} outputLength Output length (# of characters).
 * @return {tf.Model} A compiled model instance.
 */
export function createModel(
    inputVocabSize, outputVocabSize, inputLength, outputLength) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  const encoderInput = tf.input({shape: [inputLength]});
  const decoderInput = tf.input({shape: [outputLength]});

  let encoder = tf.layers.embedding({
    inputDim: inputVocabSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  }).apply(encoderInput);
  encoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(encoder);

  const encoderLast = new GetLastTimestepLayer({
    name: 'encoderLast'
  }).apply(encoder);

  let decoder = tf.layers.embedding({
    inputDim: outputVocabSize,
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
  const decoderCombinedContext =
      tf.layers.concatenate().apply([context, decoder]);
  let output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: lstmUnits,
      activation: 'tanh'
    })
  }).apply(decoderCombinedContext);
  output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: outputVocabSize,
      activation: 'softmax'
    })
  }).apply(output);

  const model = tf.model({
    inputs: [encoderInput, decoderInput],
    outputs: output
  });
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });
  return model;
}

/**
 * Perform sequence-to-sequence decoding for date conversion.
 *
 * @param {tf.Model} model The model to be used for the sequence-to-sequence
 *   decoding, with two inputs:
 *   1. Encoder input of shape `[numExamples, inputLength]`
 *   2. Decoder input of shape `[numExamples, outputLength]`
 *   and one output:
 *   1. Decoder softmax probability output of shape
 *      `[numExamples, outputLength, outputVocabularySize]`
 * @param {string} inputStr Input date string to be converted.
 * @return {{outputStr: string, attention?: tf.Tensor}}
 *   - The `outputStr` field is the output date string.
 *   - If and only if `getAttention` is `true`, the `attention` field will
 *     be populated by attention matrix as a `tf.Tensor` of
 *     dtype `float32` and shape `[]`.
 */
export async function runSeq2SeqInference(
    model, inputStr, getAttention = false) {
  return tf.tidy(() => {
    const encoderInput = dateFormat.encodeInputDateStrings([inputStr]);
    const decoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
    decoderInput.set(dateFormat.START_CODE, 0, 0);

    for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
      const predictOut = model.predict(
          [encoderInput, decoderInput.toTensor()]);
      const output = predictOut.argMax(2).dataSync()[i - 1];
      predictOut.dispose();
      decoderInput.set(output, 0, i);
    }

    const output = {outputStr: ''};

    // The `tf.Model` instance used for the final time step varies depending on
    // whether the attention matrix is requested or not.
    let finalStepModel = model;
    if (getAttention) {
      // If the attention matrix is requested, construct a two-output model.
      // - The 1st output is the original decoder output.
      // - The 2nd output is the attention matrix.
      finalStepModel = tf.model({
        inputs: model.inputs,
        outputs: model.outputs.concat([model.getLayer('attention').output])
      });
    }

    const finalPredictOut = finalStepModel.predict(
        [encoderInput, decoderInput.toTensor()]);
    let decoderFinalOutput;  // The decoder's final output.
    if (getAttention) {
      decoderFinalOutput = finalPredictOut[0];
      output.attention = finalPredictOut[1];
    } else {
      decoderFinalOutput = finalPredictOut;
    }
    decoderFinalOutput =
    decoderFinalOutput.argMax(2).dataSync()[dateFormat.OUTPUT_LENGTH - 1];

    for (let i = 1; i < decoderInput.shape[1]; ++i) {
      output.outputStr += dateFormat.OUTPUT_VOCAB[decoderInput.get(0, i)];
    }
    output.outputStr += dateFormat.OUTPUT_VOCAB[decoderFinalOutput];
    return output;
  });
}
