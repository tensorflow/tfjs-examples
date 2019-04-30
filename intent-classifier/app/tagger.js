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

import * as useLoader from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';

tf.ENV.set('WEBGL_PACK', false);

const EMBEDDING_DIM = 512;

let use;
/**
 * Load the universal sentence encoder model
 */
async function loadUSE() {
  if (use == null) {
    use = await useLoader.load();
  }
  return use;
}

const modelUrls = {
  'bidirectional-lstm': './models/bidirectional-tagger/model.json',
  'lstm': './models/lstm-tagger/model.json',
  'dense': './models/dense-tagger/model.json',
};

const taggers = {};
/**
 * Load a custom trained token tagger model.
 * @param {string} name Type of model to load. Should be a key in modelUrls
 */
async function loadTagger(name) {
  if (taggers[name] == null) {
    const url = modelUrls[name];
    try {
      taggers[name] = await tf.loadLayersModel(url);
      document.getElementById(name).disabled = false;
    } catch (e) {
      // Could not load that model. This is not necessarily an error
      // as the user may not have trained all the available model types
      console.log(`Could not load "${name}" model`);
    }
  }
  return taggers[name];
}

/**
 * Load metadata for a model.
 * @param {string} name Name of model. Should be a key in modelUrls
 */
async function loadMetadata(name) {
  const metadataUrl =
      modelUrls[name].replace('model.json', 'tagger_metadata.json');
  try {
    const resp = await fetch(metadataUrl);
    return resp.json();
  } catch (e) {
    // Could not load that model. This is not necessarily an error
    // as the user may not have trained all the available model types
    console.log(`Could not load "${name}" metadata`);
  }
}

/**
 * Load a number of models to allow the browser to cache them.
 */
async function loadModels() {
  const modelLoadPromises = Object.keys(modelUrls).map(loadTagger);
  return await Promise.all([loadUSE(), ...modelLoadPromises]);
}

/**
 * Split an input string into tokens, we use the same tokenization function
 * as we did during training.
 * @param {string} input
 *
 * @return {string[]}
 */
function tokenizeSentence(input) {
  return input.split(/\b/).map(t => t.trim()).filter(t => t.length !== 0);
}

/**
 * Tokenize a sentence and tag the tokens.
 *
 * @param {string} sentence sentence to tag
 * @param {string} model name of model to use
 *
 * @return {Object} dictionary of tokens, model outputs and embeddings
 */
async function tagTokens(sentence, model = 'bidirectional-lstm') {
  const [use, tagger, metadata] =
      await Promise.all([loadUSE(), loadTagger(model), loadMetadata(model)]);
  const {labels, sequenceLength} = metadata;


  let tokenized = tokenizeSentence(sentence);
  if (tokenized.length > sequenceLength) {
    console.warn(
        `Input sentence has more tokens than max allowed tokens ` +
        `(${sequenceLength}). Extra tokens will be dropped.`);
  }
  tokenized = tokenized.slice(0, sequenceLength);
  const activations = await use.embed(tokenized);

  // get prediction
  const prediction = tf.tidy(() => {
    // Make an input tensor of [1, sequence_len, embedding_size];
    const toPad = sequenceLength - tokenized.length;

    const padTensors = tf.ones([toPad, EMBEDDING_DIM]);
    const padded = activations.concat(padTensors);

    const batched = padded.expandDims();
    return tagger.predict(batched);
  });

  // Prediction data
  let predsArr = (await prediction.array())[0];

  // Add padding 'tokens' to the end of the values that will be displayed
  // in the UI. These are there for illustration.
  if (tokenized.length < sequenceLength) {
    tokenized.push(labels[2]);
    predsArr = predsArr.slice(0, tokenized.length);
  }

  // Add an extra activation to illustrate the padding inputs in the UI.
  // This is added for illustration.
  const displayActivations =
      tf.tidy(() => activations.concat(tf.ones([1, EMBEDDING_DIM])));
  const displayActicationsArr = await displayActivations.array();

  tf.dispose([activations, prediction, displayActivations]);

  return {
    tokenized: tokenized,
    tokenScores: predsArr,
    tokenEmbeddings: displayActicationsArr,
  };
}

/**
 * Render the tokens
 *
 * @param {string[]} tokens the tokens
 * @param {Array.number[]} tokenScores model scores for each token
 * @param {Array.number[]} tokenEmbeddings token embeddings
 * @param {string} model name of model
 */
async function displayTokenization(
    tokens, tokenScores, tokenEmbeddings, model) {
  const resultsDiv = document.createElement('div');
  resultsDiv.classList = `tagging`;
  resultsDiv.innerHTML = `<p class="model-type ${model}">${model}</p>`;

  displayTokens(tokens, resultsDiv);
  displayEmbeddingsPlot(tokenEmbeddings, resultsDiv);
  displayTags(tokenScores, resultsDiv, model);

  document.getElementById('taggings').appendChild(resultsDiv);
}

/**
 * Render the tokens.
 *
 * @param {string[]} tokens tokens to display
 * @param {HTMLElement} parentEl parent element
 */
function displayTokens(tokens, parentEl) {
  const tokensDiv = document.createElement('div');
  tokensDiv.classList = `tokens`;
  tokensDiv.innerHTML =
      tokens.map(token => `<div class="token">${token}</div>`).join('\n');
  parentEl.appendChild(tokensDiv);
}

const embeddingCol =
    d3.scaleSequential(d3.interpolateSpectral).domain([-0.075, 0.075]);
embeddingCol.clamp(true);

/**
 * Display an illustrative representation of the embeddings values
 * @param {*} embeddings
 * @param {*} parentEl
 */
function displayEmbeddingsPlot(embeddings, parentEl) {
  const embeddingDiv = document.createElement('div');
  embeddingDiv.classList = `embeddings`;

  embeddingDiv.innerHTML =
      embeddings
          .map(embedding => {
            // Note that this slice is arbitraty as the plot is only meant to
            // be illustrative.
            const embeddingValDivs = embedding.slice(0, 340).map(val => {
              return `<div class="embVal" ` +
                  `style="background-color:${embeddingCol(val)} "` +
                  `title="${val}"` +
                  `></div>`;
            });

            return `<div class="embedding">` +
                `${embeddingValDivs.join('\n')}</div>`;
          })
          .join('\n');

  parentEl.appendChild(embeddingDiv);
}

/**
 *
 * @param {*} tokenScores
 * @param {*} parentEl
 * @param {*} modelName
 */
async function displayTags(tokenScores, parentEl, modelName) {
  const metadata = await loadMetadata(modelName);
  const {labels} = metadata;

  const tagsDiv = document.createElement('div');
  tagsDiv.classList = `tags`;

  tagsDiv.innerHTML =
      tokenScores
          .map(scores => {
            const maxIndex = scores.indexOf(Math.max(...scores));
            const token = labels[maxIndex];
            const tokenScore = (scores[maxIndex] * 100).toPrecision(3);
            return `<div class="tag ${token}">` +
                `&nbsp;&nbsp;${token.replace(/__/g, '')}<sup>` +
                `${tokenScore}%</sup></div>`;
          })
          .join('\n');
  parentEl.appendChild(tagsDiv);
}

async function onSendMessage(inputText, model) {
  if (inputText != null && inputText.length > 0) {
    const result = await tagTokens(inputText, model);
    const {tokenized, tokenScores, tokenEmbeddings} = result;
    displayTokenization(tokenized, tokenScores, tokenEmbeddings, model);
  }
}

function setupListeners() {
  const form = document.getElementById('textentry');
  const textbox = document.getElementById('textbox');
  const modelSelect = document.getElementById('model-select');
  form.addEventListener('submit', event => {
    event.preventDefault();
    event.stopPropagation();

    const inputText = textbox.value;
    const model = modelSelect.options[modelSelect.selectedIndex].value;


    onSendMessage(inputText, model);
    textbox.value = '';
  }, false);
}

window.addEventListener('load', function() {
  setupListeners();
  loadModels();
  warmup();
});


async function warmup() {
  tagTokens('What is the weather in Cambridge MA?', 'bidirectional-lstm');
}
