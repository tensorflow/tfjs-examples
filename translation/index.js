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

let model;

async function loadHostedPretrainedModel() {
  const HOSTED_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-models/tfjs-layers/translation_en_fr_v1/model.json';
  status('Loading pretrained model from ' + HOSTED_MODEL_JSON_URL);
  try {
    model = await tf.loadModel(HOSTED_MODEL_JSON_URL);
    status('Done loading pretrained model.');
  } catch (err) {
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata() {
  return await (
             await fetch(
                 'https://storage.googleapis.com/tfjs-models/tfjs-layers/translation_en_fr_v1/metadata.json'))
      .json();
}

async function decodeSequence(inputSeq) {
  // Encode the inputs state vectors.
  let statesValue = await encoderModel.predict(inputSeq);

  // Generate empty target sequence of length 1.
  let targetSeq = tf.buffer([1, 1, numDecoderTokens]);
  // Populate the first character of the target sequence with the start
  // character.
  targetSeq.set(1, 0, 0, targetTokenIndex['\t']);

  // Sample loop for a batch of sequences.
  // (to simplfy, here we assume that a batch of size 1).
  let stopCondition = false;
  let decodedSentence = '';
  while (!stopCondition) {
    const predictOutputs =
        await decoderModel.predict([targetSeq.toTensor()].concat(statesValue));
    const outputTokens = predictOutputs[0];
    const h = predictOutputs[1];
    const c = predictOutputs[2];

    // Sample a token.
    // TODO(cais): Replace the following with tf.argmax when
    //   it is available.
    const outputTokenShape = outputTokens.shape;
    const logits = tf.sliceAlongFirstAxis(
                         tf.sliceAlongFirstAxis(outputTokens, 0, 1).reshape([
                           outputTokenShape[1], outputTokenShape[2]
                         ]),
                         outputTokenShape[1] - 1, 1)
                       .dataSync();
    let maxLogit;
    let sampledTokenIndex;
    for (let i = 0; i < logits.length; ++i) {
      if (i === 0 || logits[i] > maxLogit) {
        maxLogit = logits[i];
        sampledTokenIndex = i;
      }
    }
    const sampledChar = reverseTargetCharIndex[sampledTokenIndex];
    decodedSentence += sampledChar;

    // Exit condition: either hit max length or find stop character.
    if (sampledChar === '\n' || decodedSentence.length > maxDecoderSeqLength) {
      stopCondition = true;
    }

    // Update the target sequence (of length 1).
    targetSeq = tf.buffer([1, 1, numDecoderTokens]);
    targetSeq.set(1, 0, 0, sampledTokenIndex);

    // Update states.
    statesValue = [h, c];
  }

  return decodedSentence;
}

// Encode a string (e.g., a sentence) as a Tensor3D that can be fed directly
// into the Keras model.
function encodeString(str) {
  const strLen = str.length;
  const encoded = tf.buffer([1, maxEncoderSeqLength, numEncoderTokens]);
  for (let i = 0; i < strLen; ++i) {
    if (i >= maxEncoderSeqLength) {
      console.error(
          'Input sentence exceeds maximum encoder sequence length: ' +
          maxEncoderSeqLength);
    }
    const tokenIndex = inputTokenIndex[str[i]];
    if (tokenIndex == null) {
      console.error(
          'Character not found in input token index: "' + tokenIndex + '"');
    }
    console.log('"' + str[i] + '" --> ' + tokenIndex);
    encoded.set(1, 0, i, tokenIndex);
  }
  return encoded.toTensor();
}

async function doTranslation() {
  const inputSentence = $('#englishSentence').val();
  const inputSeq = encodeString(inputSentence);
  const decodedSentence = await decodeSequence(inputSeq);
  $('#frenchSentence').val(decodedSentence);
}

async function translation() {
  const translationMetadata = loadHostedMetadata();
  await loadHostedPretrainedModel();

  const maxDecoderSeqLength = translationMetadata['max_decoder_seq_length'];
  const maxEncoderSeqLength = translationMetadata['max_encoder_seq_length'];
  console.log('maxDecoderSeqLength = ' + maxDecoderSeqLength);
  console.log('maxEncoderSeqLength = ' + maxEncoderSeqLength);
  const inputTokenIndex = translationMetadata['input_token_index'];
  const targetTokenIndex = translationMetadata['target_token_index'];

  const reverseInputCharIndex = _.invert(inputTokenIndex);
  const reverseTargetCharIndex = _.invert(targetTokenIndex);

  console.log('Loading model...');
  const model = await tfjs_layers.loadModel(artifactsDir + 'model.json');
  console.log('Done loading model.');

  const numEncoderTokens = model.input[0].shape[2];
  const numDecoderTokens = model.input[1].shape[2];
  console.log('numEncoderTokens = ' + numEncoderTokens);
  console.log('numDecoderTokens = ' + numDecoderTokens);

  const encoderInputs = model.input[0];
  console.assert(model.layers[2].constructor.name === 'LSTM');
  const stateH = model.layers[2].output[1];
  const stateC = model.layers[2].output[2];
  const encoderStates = [stateH, stateC];

  const encoderModel =
      tf.model({inputs: encoderInputs, outputs: encoderStates});

  const latentDim = stateH.shape[stateH.shape.length - 1];
  console.log('latentDim = ' + latentDim);
  const decoderStateInputH =
      tf.input({shape: [latentDim], name: 'decoder_state_input_h'});
  const decoderStateInputC =
      tf.input({shape: [latentDim], name: 'decoder_state_input_c'});
  const decoderStateInputs = [decoderStateInputH, decoderStateInputC];

  console.assert(model.layers[3].constructor.name === 'LSTM');
  const decoderLSTM = model.layers[3];
  const decoderInputs = decoderLSTM.input[0];
  const applyOutputs =
      decoderLSTM.apply(decoderInputs, {initialState: decoderStateInputs});
  let decoderOutputs = applyOutputs[0];
  const decoderStateH = applyOutputs[1];
  const decoderStateC = applyOutputs[2];
  const decoderStates = [decoderStateH, decoderStateC];

  const decoderDense = model.layers[4];
  console.assert(decoderDense.constructor.name === 'Dense');
  decoderOutputs = decoderDense.apply(decoderOutputs);
  const decoderModel = tf.model({
    inputs: [decoderInputs].concat(decoderStateInputs),
    outputs: [decoderOutputs].concat(decoderStates)
  });


  $('#translate').click(doTranslation);

  doTranslation();
}

translation();
