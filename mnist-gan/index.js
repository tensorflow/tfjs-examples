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

/**
 * This file runs inference on a pretrained simple object detection model.
 *
 * The model is defined and trained with `train.js`.
 * The data used for model training and model inference are synthesized
 * programmatically. See `synthetic_images.js` for details.
 */

import * as tf from '@tensorflow/tfjs';

const status = document.getElementById('status');
const loadHostedModel = document.getElementById('load-hosted-model');
const testModel = document.getElementById('test');
const canvas = document.getElementById('data-canvas');

async function generateAndVisualizeImages(model) {
  tf.util.assert(
      model.inputs.length === 2,
      `Expected model to have exactly 2 symbolic inputs, ` +  
      `but there are ${model.inputs.length}`);
  const combinedImage = tf.tidy(() => {
    const latentDims = model.inputs[0].shape[1];
    const noise = tf.randomUniform([10, latentDims]);
    const sampledLabels = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 1]);
    const generateImages = model.predict([noise, sampledLabels]).add(1).div(2);
    return tf.concat(tf.unstack(generateImages), 1);
  });
  await tf.toPixels(combinedImage, canvas);
  tf.dispose(combinedImage);
}

async function init() {
  const LOCAL_MODEL_PATH = 'generator/model.json';
  // const HOSTED_MODEL_PATH = // TODO(cais);

  // Attempt to load locally-saved model. If it fails, activate the
  // "Load hosted model" button.
  let model;
  try {
    status.textContent = `Loading model from ${LOCAL_MODEL_PATH}...`;
    model = await tf.loadModel(LOCAL_MODEL_PATH);
    model.summary();
    testModel.disabled = false;
    status.textContent = 'Loaded locally-saved model! Now click "Test Model".'
  } catch (err) {
    status.textContent = 'Failed to load locally-saved model. ' +
        'Please click "Load Hosted Model"';
    loadHostedModel.disabled = false;
  }

  loadHostedModel.addEventListener('click', async () => {
    try {
      status.textContent =
          `Loading hosted model from ${HOSTED_MODEL_PATH} ...`;
      model = await tf.loadModel(HOSTED_MODEL_PATH);
      model.summary();
      loadHostedModel.disabled = true;
      testModel.disabled = false;
      status.textContent =
          `Loaded hosted model successfully. Now click "Test Model".`;
      runAndVisualizeInference(model);
    } catch (err) {
      status.textContent =
          `Failed to load hosted model from ${HOSTED_MODEL_PATH}`;
    }
  });

  testModel.addEventListener('click', () => generateAndVisualizeImages(model));
}

init();
