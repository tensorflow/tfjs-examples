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

import {ControllerDataset} from './controller_dataset';
import {Webcam} from './webcam';

let isPredicting = false;
const NUM_CLASSES = 4;

const PACMAN_FPS = 15;
Pacman.FPS = PACMAN_FPS;

const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_CODES = ['ARROW_UP', 'ARROW_DOWN', 'ARROW_LEFT', 'ARROW_RIGHT'];

const webcamElement = document.getElementById('webcam');
const webcam = new Webcam(webcamElement);

const trainStatus = document.getElementById('train-status');

let mobilenet;
let model;

const controllerDataset = new ControllerDataset(NUM_CLASSES);

async function train() {
  trainStatus.innerHTML = 'Training...';
  await tf.nextFrame();
  await tf.nextFrame();

  isPredicting = false;
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}), tf.layers.dense({
        units: getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        useBias: true,
        inputShape: [1000]
      }),
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        kernelRegularizer: 'l1l2',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  const sgd = tf.train.adam(getLearningRate());
  model.compile({optimizer: sgd, loss: 'categoricalCrossentropy'});

  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainStatus.innerText = 'Cost: ' + logs.loss.toFixed(5);
      }
    }
  });
}

async function predict() {
  statusElement.style.visibility = 'visible';
  let lastTime = performance.now();
  while (isPredicting) {
    const prediction = tf.tidy(() => {
      const img = webcam.capture();
      const act = getActivation(img);
      return model.predict(act);
    });

    const classId = (await prediction.as1D().argMax().data())[0];
    const control = CONTROL_CODES[classId];
    fireEvent(control);

    const elapsed = performance.now() - lastTime;

    lastTime = performance.now();
    statusElement.innerText = CONTROLS[classId];
    document.getElementById('inferenceTime').innerText =
        'inference: ' + elapsed + 'ms';

    await tf.nextFrame();
  }
  statusElement.style.visibility = 'hidden';
}

function addExample(label) {
  const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
  tf.tidy(() => {
    const img = webcam.capture();
    if (thumbDisplayed[label] == null) {
      draw(img, thumbCanvas);
    }
    controllerDataset.addExample(getActivation(img), label);
  });
}

function getActivation(img) {
  return tf.tidy(() => mobilenet.predict(img.expandDims(0)));
}

async function loadMobilenet() {
  // TODO(nsthorat): Move these to GCP when they are no longer JSON.
  const model = await tf.loadModel(
      // tslint:disable-next-line:max-line-length
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = model.getLayer('conv_pw_13_relu');
  return tf.model({inputs: model.inputs, outputs: layer.output});
}

let mouseDown = false;
const totals = [0, 0, 0, 0];

const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');

const thumbDisplayed = {};

async function handler(label) {
  mouseDown = true;
  const className = CONTROLS[label];
  const button = document.getElementById(className);
  while (mouseDown) {
    addExample(label);
    button.innerText = className + ' (' + (totals[label]++) + ')';
    await tf.nextFrame();
  }
}

function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

upButton.addEventListener('mousedown', () => handler(0));
upButton.addEventListener('mouseup', () => mouseDown = false);

downButton.addEventListener('mousedown', () => handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', () => handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', () => handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

document.getElementById('train').addEventListener('click', () => train());
document.getElementById('predict').addEventListener('click', () => {
  startPacman();
  isPredicting = true;
  predict();
});

// Set hyper params from values above.
const learningRateElement = document.getElementById('learningRate');
const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs');
const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById('dense-units');
const getDenseUnits = () => +denseUnitsElement.value;
const statusElement = document.getElementById('status');

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  // Warm up the model.
  tf.tidy(() => getActivation(webcam.capture()));

  // Show the controls once everything has loaded.
  document.getElementById('controls').style.display = '';
  document.getElementsByClassName('train-container')[0].style.visibility =
      'visible';
  document.getElementById('cost-container').style.visibility = 'visible';
  statusElement.style.visibility = 'hidden';
}

const pacmanElement = document.getElementById('pacman');

function startPacman() {
  fireEvent('N');
}
function fireEvent(keyCode) {
  const e = new KeyboardEvent('keydown');

  Object.defineProperty(e, 'keyCode', {
    get: () => {
      return KEY[keyCode];
    }
  });

  pacmanElement.dispatchEvent(e);
  document.dispatchEvent(e);
}

PACMAN.init(
    pacmanElement,
    'http://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/');

// Initialize the application.
init();
