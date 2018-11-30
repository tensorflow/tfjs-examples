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

import {loadMnistData, sampleFromMnistData} from './web-data';

const status = document.getElementById('status');
// const loadHostedModel = document.getElementById('load-hosted-model');
const testModel = document.getElementById('test');
const zSpaceSpan = document.getElementById('z-space-span');
const fakeImagesSpan = document.getElementById('fake-images-span');
const fakeCanvas = document.getElementById('fake-canvas');
const realCanvas = document.getElementById('real-canvas');

/**
 * Generate values for the latent vector and show them with the sliders.
 *
 * @param {bool} fixedLatent Whether to use fixed value for the latent
 *   vector (0.5 for every dimension).
 */
function generateLatentVector(fixedLatent) {
  return tf.tidy(() => {
    const latentDims = latentSliders.length;

    // Generate random latent vector (a.k.a, z-space vector).
    const latentValues = [];
    for (let i = 0; i < latentDims; ++i) {
      const latentValue = fixedLatent === true ? 0.5 : Math.random();
      latentValues.push(latentValue);
      latentSliders[i].value = latentValue;
    }
  });
}

/**
 * Read the value of the latent-space vector fromthe sliders.
 *
 * @param {number} numRepeats Number of times to tile the single latent vector
 *   for. Used for generating a batch of fake MNIST images.
 * @returns The tiled latent-space vector, of shape [numRepeats, latentDim].
 */
function getLatentVectors(numRepeats) {
  return tf.tidy(() => {
    const latentDims = latentSliders.length;
    const zs = [];
    for (let i = 0; i < latentDims; ++i) {
      zs.push(latentSliders[i].value);
    }
    const singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
    return singleLatentVector.tile([numRepeats, 1]);
  });
}

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

  const combinedFakes = tf.tidy(() => {
    const latentVectors = getLatentVectors(10);

    // Generate one fake image for each digit.
    const sampledLabels = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 1]);
    // The output has pixel values in the [-1, 1] interval. Normalize it
    // to the unit inerval ([0, 1]).
    const t0 = tf.util.now();
    const generateImages =
        generator.predict([latentVectors, sampledLabels]).add(1).div(2);
    const elapsed = tf.util.now() - t0;
    fakeImagesSpan.textContent =
        `Fake images (generation took ${elapsed.toFixed(2)} ms)`;
    // Concatenate the images horizontally into a single image.
    return tf.concat(tf.unstack(generateImages), 1);
  });

  await tf.toPixels(combinedFakes, fakeCanvas);
  tf.dispose(combinedFakes);
}

/** Refresh examples of real MNIST images. */
async function drawReals() {
  const combinedReals = sampleFromMnistData(10);
  await tf.toPixels(combinedReals, realCanvas);
  tf.dispose(combinedReals);
}

/** An array that holds all sliders for the latent-space values. */
let latentSliders;

/**
 * Create sliders for the latent space.
 *
 * @param {tf.Model} generator The generator part of the trained ACGAN.
 */
function createSliders(generator) {
  const latentDims = generator.inputs[0].shape[1];
  latentSliders = [];
  const slidersContainer = document.getElementById('sliders-container');
  for (let i = 0; i < latentDims; ++i) {
    const slider = document.createElement('input');
    slider.setAttribute('type', 'range');
    slider.min = 0;
    slider.max = 1;
    slider.step = 0.01;
    slider.value = 0.5;
    slider.addEventListener('change', () => {
      generateAndVisualizeImages(generator);
    });

    slidersContainer.appendChild(slider);
    latentSliders.push(slider);
  }
  zSpaceSpan.textContent = `z-space vector (${latentDims} dimensions)`;
}

async function init() {
  // Load MNIST data for display in webpage.
  status.textContent = 'Loading MNIST data...';
  await loadMnistData();

  const LOCAL_MEATADATA_PATH = 'generator/acgan-metadata.json';
  const LOCAL_MODEL_PATH = 'generator/model.json';
  // TODO(cais): Add URL to hosted model and logic for loading it.
  // const HOSTED_MODEL_PATH = '';

  // Attempt to load locally-saved model. If it fails, activate the
  // "Load hosted model" button.
  let model;
  try {
    status.textContent = 'Loading metadata';
    const metadata =
        await (await fetch(LOCAL_MEATADATA_PATH, {cache: 'no-cache'})).json();

    status.textContent = `Loading model from ${LOCAL_MODEL_PATH}...`;
    model = await tf.loadModel(
        tf.io.browserHTTPRequest(LOCAL_MODEL_PATH, {cache: 'no-cache'}));
    model.summary();

    // Create slider for the z-space (latent vectors).
    createSliders(model);

    testModel.disabled = false;
    if (metadata.completed) {
      status.textContent =
          `Training of ACGAN in Node.js (${metadata.totalEpochs} epochs) ` +
          `is completed. `;
    } else {
      status.textContent = `Training of ACGAN in Node.js is ongoing (epoch ` +
          `${metadata.currentEpoch + 1}/${metadata.totalEpochs})... `;
    }
    status.textContent +=
        'Loaded locally-saved model! Now click "Generate" or ' +
        'adjust the z-space sliders.';

    generateLatentVector(true);
    generateAndVisualizeImages(model);
    drawReals();
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

  testModel.addEventListener('click', () => {
    generateLatentVector(false);
    generateAndVisualizeImages(model);
    drawReals();
  });
}

init();
