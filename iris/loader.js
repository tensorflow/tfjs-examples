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
import * as ui from './ui';

/**
 * Test whether a given URL is retrievable.
 */
export async function urlExists(url) {
  ui.status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

/**
 * Load pretrained model stored at a remote URL.
 *
 * @return An instance of `tf.Model` with model topology and weights loaded.
 */
export async function loadHostedPretrainedModel(url) {
  ui.status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    ui.status('Done loading pretrained model.');
    return model;
  } catch (err) {
    console.error(err);
    ui.status('Loading pretrained model failed.');
  }
}

// The URL-like path that identifies the client-side location where downloaded
// or locally trained models can be stored.
const LOCAL_MODEL_URL = 'indexeddb://tfjs-iris-demo-model/v1';

export async function saveModelLocally(model) {
  const saveResult = await model.save(LOCAL_MODEL_URL);
}

export async function loadModelLocally() {
  return await tf.loadLayersModel(LOCAL_MODEL_URL);
}

export async function removeModelLocally() {
  return await tf.io.removeModel(LOCAL_MODEL_URL);
}

/**
 * Check the presence and status of locally saved models (e.g., in IndexedDB).
 *
 * Update the UI control states accordingly.
 */
export async function updateLocalModelStatus() {
  const localModelStatus = document.getElementById('local-model-status');
  const localLoadButton = document.getElementById('load-local');
  const localRemoveButton = document.getElementById('remove-local');

  const modelsInfo = await tf.io.listModels();
  if (LOCAL_MODEL_URL in modelsInfo) {
    localModelStatus.textContent = 'Found locally-stored model saved at ' +
        modelsInfo[LOCAL_MODEL_URL].dateSaved.toDateString();
    localLoadButton.disabled = false;
    localRemoveButton.disabled = false;
  } else {
    localModelStatus.textContent = 'No locally-stored model is found.';
    localLoadButton.disabled = true;
    localRemoveButton.disabled = true;
  }
}
