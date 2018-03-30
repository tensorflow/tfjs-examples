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
import * as ui from './ui';
import {Webcam} from './webcam';

let isPredicting = false;
const NUM_CLASSES = 4;

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
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  const sgd = tf.train.adam(ui.getLearningRate());
  model.compile({optimizer: sgd, loss: 'categoricalCrossentropy'});

  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainStatus.innerText = 'Loss: ' + logs.loss.toFixed(5);
        await tf.nextFrame();
      }
    }
  });
}

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const prediction = tf.tidy(() => {
      const img = webcam.capture();
      const act = getActivation(img);
      return model.predict(act);
    });

    const classId = (await prediction.as1D().argMax().data())[0];

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(getActivation(img), label);

    ui.drawThumb(img, label);
  });
});

function getActivation(img) {
  return tf.tidy(() => mobilenet.predict(img.expandDims(0)));
}

async function loadMobilenet() {
  const model = await tf.loadModel(
      // tslint:disable-next-line:max-line-length
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = model.getLayer('conv_pw_13_relu');
  return tf.model({inputs: model.inputs, outputs: layer.output});
}

document.getElementById('train').addEventListener('click', () => train());
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  // Warm up the model.
  tf.tidy(() => getActivation(webcam.capture()));

  ui.init();
}

// Initialize the application.
init();
