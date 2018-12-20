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
import * as loader from './loader';
import * as ui from './ui';


const HOSTED_URLS = {
  model:
      'https://storage.googleapis.com/tfjs-models/tfjs/translation_en_fr_v1/model.json',
  metadata:
      'https://storage.googleapis.com/tfjs-models/tfjs/translation_en_fr_v1/metadata.json'
};

const LOCAL_URLS = {
  model: 'http://localhost:1235/resources/model.json',
  metadata: 'http://localhost:1235/resources/metadata.json'
};

class Translator {
  /**
   * Initializes the Translation demo.
   */
  async init(urls) {
    this.urls = urls;
    const model = await loader.loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    this.prepareEncoderModel(model);
    this.prepareDecoderModel(model);
    return this;
  }

  async loadMetadata() {
    const translationMetadata =
        await loader.loadHostedMetadata(this.urls.metadata);
    this.maxDecoderSeqLength = translationMetadata['max_decoder_seq_length'];
    this.maxEncoderSeqLength = translationMetadata['max_encoder_seq_length'];
    console.log('maxDecoderSeqLength = ' + this.maxDecoderSeqLength);
    console.log('maxEncoderSeqLength = ' + this.maxEncoderSeqLength);
    this.inputTokenIndex = translationMetadata['input_token_index'];
    this.targetTokenIndex = translationMetadata['target_token_index'];
    this.reverseTargetCharIndex =
        Object.keys(this.targetTokenIndex)
            .reduce(
                (obj, key) => (obj[this.targetTokenIndex[key]] = key, obj), {});
  }

  prepareEncoderModel(model) {
    this.numEncoderTokens = model.input[0].shape[2];
    console.log('numEncoderTokens = ' + this.numEncoderTokens);

    const encoderInputs = model.input[0];
    const stateH = model.layers[2].output[1];
    const stateC = model.layers[2].output[2];
    const encoderStates = [stateH, stateC];

    this.encoderModel =
        tf.model({inputs: encoderInputs, outputs: encoderStates});
  }

  prepareDecoderModel(model) {
    this.numDecoderTokens = model.input[1].shape[2];
    console.log('numDecoderTokens = ' + this.numDecoderTokens);

    const stateH = model.layers[2].output[1];
    const latentDim = stateH.shape[stateH.shape.length - 1];
    console.log('latentDim = ' + latentDim);
    const decoderStateInputH =
        tf.input({shape: [latentDim], name: 'decoder_state_input_h'});
    const decoderStateInputC =
        tf.input({shape: [latentDim], name: 'decoder_state_input_c'});
    const decoderStateInputs = [decoderStateInputH, decoderStateInputC];

    const decoderLSTM = model.layers[3];
    const decoderInputs = decoderLSTM.input[0];
    const applyOutputs =
        decoderLSTM.apply(decoderInputs, {initialState: decoderStateInputs});
    let decoderOutputs = applyOutputs[0];
    const decoderStateH = applyOutputs[1];
    const decoderStateC = applyOutputs[2];
    const decoderStates = [decoderStateH, decoderStateC];

    const decoderDense = model.layers[4];
    decoderOutputs = decoderDense.apply(decoderOutputs);
    this.decoderModel = tf.model({
      inputs: [decoderInputs].concat(decoderStateInputs),
      outputs: [decoderOutputs].concat(decoderStates)
    });
  }

  /**
   * Encode a string (e.g., a sentence) as a Tensor3D that can be fed directly
   * into the TensorFlow.js model.
   */
  encodeString(str) {
    const strLen = str.length;
    const encoded =
        tf.buffer([1, this.maxEncoderSeqLength, this.numEncoderTokens]);
    for (let i = 0; i < strLen; ++i) {
      if (i >= this.maxEncoderSeqLength) {
        console.error(
            'Input sentence exceeds maximum encoder sequence length: ' +
            this.maxEncoderSeqLength);
      }

      const tokenIndex = this.inputTokenIndex[str[i]];
      if (tokenIndex == null) {
        console.error(
            'Character not found in input token index: "' + tokenIndex + '"');
      }
      encoded.set(1, 0, i, tokenIndex);
    }
    return encoded.toTensor();
  }

  decodeSequence(inputSeq) {
    // Encode the inputs state vectors.
    let statesValue = this.encoderModel.predict(inputSeq);

    // Generate empty target sequence of length 1.
    let targetSeq = tf.buffer([1, 1, this.numDecoderTokens]);
    // Populate the first character of the target sequence with the start
    // character.
    targetSeq.set(1, 0, 0, this.targetTokenIndex['\t']);

    // Sample loop for a batch of sequences.
    // (to simplify, here we assume that a batch of size 1).
    let stopCondition = false;
    let decodedSentence = '';
    while (!stopCondition) {
      const predictOutputs =
          this.decoderModel.predict([targetSeq.toTensor()].concat(statesValue));
      const outputTokens = predictOutputs[0];
      const h = predictOutputs[1];
      const c = predictOutputs[2];

      // Sample a token.
      // We know that outputTokens.shape is [1, 1, n], so no need for slicing.
      const logits = outputTokens.reshape([outputTokens.shape[2]]);
      const sampledTokenIndex = logits.argMax().dataSync()[0];
      const sampledChar = this.reverseTargetCharIndex[sampledTokenIndex];
      decodedSentence += sampledChar;

      // Exit condition: either hit max length or find stop character.
      if (sampledChar === '\n' ||
          decodedSentence.length > this.maxDecoderSeqLength) {
        stopCondition = true;
      }

      // Update the target sequence (of length 1).
      targetSeq = tf.buffer([1, 1, this.numDecoderTokens]);
      targetSeq.set(1, 0, 0, sampledTokenIndex);

      // Update states.
      statesValue = [h, c];
    }

    return decodedSentence;
  }

  /** Translate the given English sentence into French. */
  translate(inputSentence) {
    const inputSeq = this.encodeString(inputSentence);
    const decodedSentence = this.decodeSequence(inputSeq);
    return decodedSentence;
  }
}

/**
 * Loads the pretrained model and metadata, and registers the translation
 * function with the UI.
 */
async function setupTranslator() {
  if (await loader.urlExists(HOSTED_URLS.model)) {
    ui.status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      const translator = await new Translator().init(HOSTED_URLS);
      ui.setTranslationFunction(x => translator.translate(x));
      ui.setEnglish('Go.', x => translator.translate(x));
    });
    button.style.display = 'inline-block';
  }

  if (await loader.urlExists(LOCAL_URLS.model)) {
    ui.status('Model available: ' + LOCAL_URLS.model);
    const button = document.getElementById('load-pretrained-local');
    button.addEventListener('click', async () => {
      const translator = await new Translator().init(LOCAL_URLS);
      ui.setTranslationFunction(x => translator.translate(x));
      ui.setEnglish('Go.', x => translator.translate(x));
    });
    button.style.display = 'inline-block';
  }

  ui.status('Standing by.');
}

setupTranslator();
