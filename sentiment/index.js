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

const HOSTED_MODEL_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json';
const HOSTED_METADATA_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json';

class SentimentPredictor {
  /**
   * Initializes the Sentiment demo.
   */
  async init() {
    this.model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const sentimentMetadata =
        await loader.loadHostedMetadata(HOSTED_METADATA_JSON_URL);
    ui.showMetadata(sentimentMetadata);
    this.indexFrom = sentimentMetadata['index_from'];
    this.maxLen = sentimentMetadata['max_len'];
    console.log('indexFrom = ' + this.indexFrom);
    console.log('maxLen = ' + this.maxLen);

    this.wordIndex = sentimentMetadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    ui.status(inputText);
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      // TODO(cais): Deal with OOV words.
      const word = inputText[i];
      ui.status(word);
      inputBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i);
    }
    const input = inputBuffer.toTensor();

    ui.status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};


/**
 * Loads the pretrained model and metadata, and registers the predict
 * function with the UI.
 */
async function setupSentiment() {
  const predictor = await new SentimentPredictor().init();
  ui.setPredictFunction(x => predictor.predict(x));
  ui.prepUI(x => predictor.predict(x));
}

setupSentiment();
