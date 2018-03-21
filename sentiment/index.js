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

function status(statusText) {
  document.getElementById('status').textContent = statusText;
}

/**
 * Load pretrained model stored at a remote URL.
 *
 * @return An instance of `tf.Model` with model topology and weights loaded.
 */
async function loadHostedPretrainedModel() {
  const HOSTED_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json';
  status('Loading pretrained model from ' + HOSTED_MODEL_JSON_URL);
  try {
    model = await tf.loadModel(HOSTED_MODEL_JSON_URL);
    status('Done loading pretrained model.');
  } catch (err) {
    console.log(err);
    status('Loading pretrained model failed.');
  }
}

/**
 * Load metadata file stored at a remote URL.
 *
 * @return An object containing metadata as key-value pairs.
 */
async function loadHostedMetadata() {
  const HOSTED_METADATA_JSON_URL =
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json';
  status('Loading metadata from ' + HOSTED_METADATA_JSON_URL);
  try {
    const metadataJson = await fetch(HOSTED_METADATA_JSON_URL);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.log(err);
    status('Loading metadata failed.');
  }
}

/**
 * The main function of the Sentiment demo.
 *
 * Loads the pretrained model and metadata, and registers a listener to run
 * inference on the contents of the text box when a button is clicked.
 */
async function sentiment() {
  const sentimentMetadataJSON = await loadHostedMetadata();
  await loadHostedPretrainedModel();

  document.getElementById('modelType').textContent =
      sentimentMetadataJSON['model_type'];
  document.getElementById('vocabularySize').textContent =
      sentimentMetadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      sentimentMetadataJSON['max_len'];

  const exampleReviews = {
    'positive':
        'die hard mario fan and i loved this game br br this game starts slightly boring but trust me it\'s worth it as soon as you start your hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i\'m not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it\'ll be worth it br br story 9 9 action 10 1 it\'s that good OOV 10 attention OOV 10 average 10',
    'negative':
        'the mother in this movie is reckless with her children to the point of neglect i wish i wasn\'t so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can\'t we have night vision'
  };
  const testExampleSelect = document.getElementById('test-example-select');
  const reviewText = document.getElementById('review-text');
  const runInference = document.getElementById('run-inference');
  testExampleSelect.addEventListener('change', () => {
    reviewText.value = exampleReviews[testExampleSelect.value];
  });
  reviewText.value = exampleReviews['positive'];

  const indexFrom = sentimentMetadataJSON['index_from'];
  const maxLen = sentimentMetadataJSON['max_len'];
  console.log('indexFrom = ' + indexFrom);
  console.log('maxLen = ' + maxLen);

  const wordIndex = sentimentMetadataJSON['word_index']

  runInference.addEventListener('click', async () => {
    // Convert to lower case and remove all punctuations.
    const inputText = reviewText.value.trim()
                          .toLowerCase()
                          .replace(/(\.|\,|\!)/g, '')
                          .split(' ');
    status(inputText);
    // Look up word indices.
    const inputBuffer = tf.buffer([1, maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      // TODO(cais): Deal with OOV words.
      const word = inputText[i];
      status(word);
      inputBuffer.set(wordIndex[word] + indexFrom, 0, i);
    }
    const input = inputBuffer.toTensor();
    console.log('inputBuffer.values:', inputBuffer.values);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = performance.now();

    status(
        'Inference result (0 - negative; 1 - positive): ' + score +
        ' (elapsed: ' + (endMs - beginMs) + ' ms)');
  });
}

sentiment();
