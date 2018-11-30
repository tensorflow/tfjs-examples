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
 * This file loads checkpoints saved from gan.js during the trainig of an ACGAN
 * and demonstrates the generated fake MNIST images.
 */

import * as tf from '@tensorflow/tfjs';

const status = document.getElementById('status');
// const loadHostedModel = document.getElementById('load-hosted-model');
const testModel = document.getElementById('test');
const canvas = document.getElementById('data-canvas');

/**
 * Generate a set of examples using the generator model of the ACGAN.
 *
 * @param {tf.Model} generator The generator part of the ACGAN.
 */
async function generateAndVisualizeImages(generator) {
  tf.util.assert(
      generator.inputs.length === 2,
      `Expected model to have exactly 2 symbolic inputs, ` +
          `but there are ${generator.inputs.length}`);

  const combinedImage = tf.tidy(() => {
    const latentDims = generator.inputs[0].shape[1];
    // Random latent vector (a.k.a, z-space vector).
    const noise = tf.randomUniform([10, latentDims]);
    // Generate one fake image for each digit.
    const sampledLabels = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 1]);
    // The output has pixel values in the [-1, 1] interval. Normalize it
    // to the unit inerval ([0, 1]).
    const generateImages =
        generator.predict([noise, sampledLabels]).add(1).div(2);
    // Concatenate the images horizontally into a single image.
    return tf.concat(tf.unstack(generateImages), 1);
  });

  await tf.toPixels(combinedImage, canvas);
  tf.dispose(combinedImage);
}

async function init() {
  const LOCAL_MEATADATA_PATH = 'generator/acgan-metadata.json';
  const LOCAL_MODEL_PATH = 'generator/model.json';
  // TODO(cais): Add URL to hosted model and logic for loading it.
  // const HOSTED_MODEL_PATH = '';

  // Attempt to load locally-saved model. If it fails, activate the
  // "Load hosted model" button.
  let model;
  try {
    console.log('Loading metadata');
    const metadata =
        await (await fetch(LOCAL_MEATADATA_PATH, {cache: 'no-cache'})).json();

    status.textContent = `Loading model from ${LOCAL_MODEL_PATH}...`;
    model = await tf.loadModel(
        tf.io.browserHTTPRequest(LOCAL_MODEL_PATH, {cache: 'no-cache'}));
    model.summary();

    testModel.disabled = false;
    if (metadata.completed) {
      status.textContent =
          `Training of ACGAN in Node.js (${metadata.totalEpochs} epochs) ` +
          `is completed. `;
    } else {
      status.textContent = `Training of ACGAN in Node.js is ongoing (epoch ` +
          `${metadata.currentEpoch + 1}/${metadata.totalEpochs}). `;
    }
    status.textContent += 'Loaded locally-saved model! Now click "Test Model".'

    generateAndVisualizeImages(model);
  } catch (err) {
    status.textContent =
        'Failed to load locally-saved model and/or metadata. ' +
        'Please click "Load Hosted Model"';
    // TODO(cais): Add logic for loading hosted generator model.
    // loadHostedModel.disabled = false;
  }

  // TODO(cais): Add logic for loading hosted generator model.
  // loadHostedModel.addEventListener('click', async () => {
  //   try {
  //     status.textContent =
  //         `Loading hosted model from ${HOSTED_MODEL_PATH} ...`;
  //     model = await tf.loadModel(HOSTED_MODEL_PATH);
  //     model.summary();
  //     loadHostedModel.disabled = true;
  //     testModel.disabled = false;
  //     status.textContent =
  //         `Loaded hosted model successfully. Now click "Test Model".`;
  //     runAndVisualizeInference(model);
  //   } catch (err) {
  //     status.textContent =
  //         `Failed to load hosted model from ${HOSTED_MODEL_PATH}`;
  //   }
  // });

  testModel.addEventListener('click', () => generateAndVisualizeImages(model));
}

init();
