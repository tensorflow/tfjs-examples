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

/**
 * This file holds the browser based viewer for the VAE trained in node.
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {select as d3Select} from 'd3-selection';

// Make sure that you are serving the model file at this location.
// You should be able to paste this url into your browser and see
// the json file.
const decoderUrl = './models/decoder/model.json';

let decoder;

const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_CHANNELS = 1;

const LATENT_DIMS = 2;

async function loadModel(modelUrl) {
  const decoder = await tf.loadLayersModel(modelUrl);

  const queryString = window.location.search.substring(1);
  if (queryString.match('debug')) {
    tfvis.show.modelSummary({name: 'decoder'}, decoder);
    tfvis.show.layer({name: 'dense2'}, decoder.getLayer('dense_Dense2'));
    tfvis.show.layer({name: 'dense3'}, decoder.getLayer('dense_Dense3'));
  }
  return decoder;
}

/**
 * Generates a representation of a latent space.
 *
 * Returns an array of tensors representing each dimension. Currently
 * each dimension is evenly spaced in the same way.
 *
 * @param {number} dimensions number of dimensions
 * @param {number} pointsPerDim number of points in each dimension
 * @param {number} start start value
 * @param {number} end end value
 * @returns {Tensor1d[]}
 */
function generateLatentSpace(dimensions, pointsPerDim, start, end) {
  const result = [];
  for (let i = 0; i < dimensions; i++) {
    const values = tf.linspace(start, end, pointsPerDim);
    result.push(values);
  }

  return result;
}

/**
 * Decode a (batch of) z vector into an image tensor. Z is the vector in latent
 * space that we want to generate an image for.
 *
 * Returns an image tensor of the shape [batch, IMAGE_HEIGHT, IMAGE_WIDTH,
 * IMAGE_CHANNELS]
 *
 * @param {Tensor2D} inputTensor of shape [batch, LATENT_DIMS]
 */
function decodeZ(inputTensor) {
  return tf.tidy(() => {
    const res = decoder.predict(inputTensor).mul(255).cast('int32');
    const reshaped = res.reshape(
        [inputTensor.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]);
    return reshaped;
  });
}

/**
 * Render the latent space by z vectors through the VAE and rendering
 * the result.
 *
 * Handles only 2D latent spaces
 */
async function renderLatentSpace(latentSpace) {
  document.getElementById('plot-area').innerText = '';
  const [xAxis, yAxis] = latentSpace;

  // Create the canvases that we will draw to.
  const xPlaceholder = Array(xAxis.shape[0]).fill(0);
  const yPlaceholder = Array(yAxis.shape[0]).fill(0);

  const rows = d3Select('.plot-area').selectAll('div.row').data(xPlaceholder);
  const rEnter = rows.enter().append('div').attr('class', 'row');
  rows.exit().remove();

  const cols = rEnter.selectAll('div.col').data(yPlaceholder);
  cols.enter()
      .append('div')
      .attr('class', 'col')
      .append('canvas')
      .attr('width', 50)
      .attr('height', 50);

  // Generate images and render them to each canvas element
  rows.merge(rEnter).each(async function(rowZ, rowIndex) {
    // Generate a batch of zVectors for each row.
    const zX = xAxis.slice(rowIndex, 1).tile(yAxis.shape);
    const zBatch = zX.stack(yAxis).transpose();
    const batchImageTensor = decodeZ(zBatch);
    const imageTensors = batchImageTensor.unstack();

    tf.dispose([zX, zBatch, batchImageTensor]);

    const cols = d3Select(this).selectAll('.col');
    cols.each(async function(colZ, colIndex) {
      const canvas = d3Select(this).select('canvas').node();
      const imageTensor = imageTensors[colIndex];

      // Render the results to the canvas
      tf.browser.toPixels(imageTensor, canvas).then(() => {
        tf.dispose([imageTensor]);
      });
    });
  });
}

function getParams() {
  const ppd = document.getElementById('pointsPerDim');
  const start = document.getElementById('start');
  const end = document.getElementById('end');

  return {
    pointsPerDim: parseInt(ppd.value), start: parseFloat(start.value),
        end: parseFloat(end.value),
  }
}

/**
 * Generate an evenly spaced 2d latent space.
 */
function draw() {
  const params = getParams();
  console.log('params', params);
  const latentSpace = generateLatentSpace(
      LATENT_DIMS, params.pointsPerDim, params.start, params.end);

  renderLatentSpace(latentSpace);
  tf.dispose(latentSpace);
}

function setupListeners() {
  document.getElementById('update').addEventListener('click', () => {
    draw();
  })
}

// Render images generated byt the VAE.
(async function run() {
  setupListeners();
  decoder = await loadModel(decoderUrl);
  draw();
})();
