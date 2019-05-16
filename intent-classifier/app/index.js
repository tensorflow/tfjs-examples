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

tf.ENV.set('WEBGL_PACK', false);

const DENSE_MODEL_URL = './models/intent/model.json';
const METADATA_URL = './models/intent/intent_metadata.json';

let use;
async function loadUSE() {
  if (use == null) {
    use = await useLoader.load();
  }
  return use;
}

let intent;
async function loadIntentClassifer(url) {
  if (intent == null) {
    intent = await tf.loadLayersModel(url);
  }
  return intent;
}

let metadata;
async function loadMetadata() {
  if (metadata == null) {
    const resp = await fetch(METADATA_URL);
    metadata = resp.json();
  }
  return metadata;
}

async function classify(sentences) {
  const [use, intent, metadata] = await Promise.all(
      [loadUSE(), loadIntentClassifer(DENSE_MODEL_URL), loadMetadata()]);

  const {labels} = metadata;
  const activations = await use.embed(sentences);

  const prediction = intent.predict(activations);

  const predsArr = await prediction.array();
  const preview = [predsArr[0].slice()];
  preview.unshift(labels);
  console.table(preview);

  tf.dispose([activations, prediction]);

  return predsArr[0];
}

const THRESHOLD = 0.90;
async function getClassificationMessage(softmaxArr) {
  const {labels} = await loadMetadata();
  const max = Math.max(...softmaxArr);
  const maxIndex = softmaxArr.indexOf(max);
  const intentLabel = labels[maxIndex];

  if (max < THRESHOLD) {
    return '¯\\_(ツ)_/¯';
  } else {
    let response;
    switch (intentLabel) {
      case 'AddToPlaylist':
        response = '💿➡️📇';
        break;
      case 'GetWeather':
        response = '⛅';
        break;
      case 'PlayMusic':
        response = '🎵🎺🎵';
        break;
      default:
        response = '?';
        break;
    }
    return response;
  }
}

async function onSendMessage(inputText) {
  if (inputText != null && inputText.length > 0) {
    // Add the input text to the chat window
    const msgId = appendMessage(inputText, 'input');
    // Classify the text
    const classification = await classify([inputText]);
    // Add the response to the chat window
    const response = await getClassificationMessage(classification);
    appendMessage(response, 'bot', msgId);
  }
}

let messageId = 0;
function appendMessage(message, sender, appendAfter) {
  const messageDiv = document.createElement('div');
  messageDiv.classList = `message ${sender}`;
  messageDiv.innerHTML = message;
  messageDiv.dataset.messageId = messageId++;

  const messageArea = document.getElementById('message-area');
  if (appendAfter == null) {
    messageArea.appendChild(messageDiv);
  } else {
    const inputMsg =
        document.querySelector(`.message[data-message-id="${appendAfter}"]`);
    inputMsg.parentNode.insertBefore(messageDiv, inputMsg.nextElementSibling);
  }

  // Scroll the message area to the bottom.
  messageArea.scroll({top: messageArea.scrollHeight, behavior: 'smooth'});

  // Return this message id so that a reply can be posted to it later
  return messageDiv.dataset.messageId;
}

function setupListeners() {
  const form = document.getElementById('textentry');
  const textbox = document.getElementById('textbox');
  form.addEventListener('submit', event => {
    event.preventDefault();
    event.stopPropagation();

    const inputText = textbox.value;
    onSendMessage(inputText);
    textbox.value = '';
  }, false);
}

function warmup() {
  classify('hello there');
}

window.addEventListener('load', function() {
  setupListeners();
  warmup();
});
