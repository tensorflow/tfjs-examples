/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import * as tsne from '@tensorflow/tfjs-tsne';
import * as d3 from 'd3';

import {MnistData} from './data';


let canvas;
let context;
let data;

const width = 800;
const height = 800;

const colors = d3.scaleOrdinal(d3.schemeCategory10);


async function loadData() {
  data = new MnistData();
  return data.load();
}

// This will actually run the tsne algorithm iteratively and display the results
async function doEmbedding(
    data, labels, numIterations, knnIter, perplexity, onIteration) {
  const embedder = tsne.tsne(data);
  await embedder.iterateKnn(knnIter);

  for (let i = 0; i < numIterations; i++) {
    await embedder.iterate(1);  // You could also do a few more iterations in
                                // between downloading the data for display
    const coordinates = await embedder.coordsArray();
    onIteration(coordinates, labels);
    await tf.nextFrame();
  }
}

// Helper function to render our data using d3 and canvas.
function renderEmbedding(coordinates, labels) {
  const x = d3.scaleLinear().range([0, width]).domain([0, 1]);
  const y = d3.scaleLinear().range([0, height]).domain([0, 1]);


  context.clearRect(0, 0, width, height);
  coordinates.forEach(function(d, i) {
    context.font = '10px sans';
    context.fillStyle = colors(parseInt(labels[i], 10));
    context.fillText(labels[i], x(d[0]), y(d[1]));
  });
}

async function initVisualization() {
  canvas = d3.select('#vis')
               .append('canvas')
               .attr('width', width)
               .attr('height', height);

  context = canvas.node().getContext('2d');
}


async function start(numPoints = 10000, tsneIter, knnIter, perplexity) {
  const IMAGE_SIZE = 28 * 28;
  const NUM_IMAGES = 10000;
  const NUM_CLASSES = 10;
  const IMG_WIDTH = 28;
  const IMG_HEIGHT = 28;

  const NEW_HEIGHT = 10;
  const NEW_WIDTH = 10;


  const [reshaped, labels] = tf.tidy(() => {
    // Convert our labels from tensor to regular array
    const labelsTensor =
        tf.tensor2d(data.testLabels, [NUM_IMAGES, NUM_CLASSES]);
    const labels = labelsTensor.argMax(1).dataSync();

    // Take the specified number of images from the test set
    // We first do a reshape into the image dimensions then slice
    // that tensor
    const images = tf.tensor4d(data.testImages, [
                       NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1
                     ]).slice([0], [numPoints]);

    // Resize the images to reduce dimensionlity of the input data
    const resized = images.resizeBilinear([NEW_HEIGHT, NEW_WIDTH]);

    // Reshape the resized images into the rank 2 tensor expected by tfjs-tnse
    const reshaped = resized.reshape([numPoints, NEW_HEIGHT * NEW_WIDTH])

    return [reshaped, labels];
  });

  // Perform and render the T-SNE
  await doEmbedding(
      reshaped, labels, tsneIter, knnIter, perplexity, renderEmbedding);

  // Dispose the input data tensor
  reshaped.dispose();
}

function initControls() {
  const startBtn = document.getElementById('start');
  startBtn.disabled = false;

  startBtn.addEventListener('click', () => {
    const numPoints = parseInt(
        document.querySelector('input[name=\'numPoints\']:checked').value, 10);

    const perplexity =
        parseInt(document.getElementById('perplexity-input').value, 10);
    const tsneIter = parseInt(document.getElementById('tsne-input').value, 10);
    const knnIter = parseInt(document.getElementById('knn-input').value, 10);

    start(numPoints, tsneIter, knnIter, perplexity);
    startBtn.innerText = 'Restart TSNE';
  })

  // Update the labels when the sliders change.
  document.querySelectorAll('input[type=range]')
      .forEach(
          (rangeEl) => rangeEl.addEventListener(
              'input',
              () => rangeEl.previousElementSibling.innerText = rangeEl.value));
}

loadData().then(() => {
  initControls();
  initVisualization();
});
