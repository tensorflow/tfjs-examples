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

import Plotly from 'plotly.js-dist';

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

export function logStatus(message) {
  statusElement.innerText = message;
}

export function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

export function showTestResults(batch, predictions, labels) {
  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    draw(image.flatten(), canvas);

    const pred = document.createElement('div');

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }
}

const lossLabelElement = document.getElementById('loss-label');
const accuracyLabelElement = document.getElementById('accuracy-label');

const lossValues = {
  train: {
    x: [],
    y: [],
    name: 'train',
    mode: 'lines',
    line: {width: 1}
  },
  validation: {
    x: [],
    y: [],
    name: 'validation',
    mode: 'lines+markers',
    line: {width: 3}
  }
};
export function plotLoss(batch, loss, set) {
  lossValues[set].x.push(batch);
  lossValues[set].y.push(loss);
  Plotly.newPlot('loss-canvas', [lossValues.train, lossValues.validation], {
    width: 480,
    xaxis: {title: 'batch #'},
    yaxis: {title: 'loss'},
    font: {size: 18}
  });
  lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
}

const accuracyValues = {
  train: {
    x: [],
    y: [],
    name: 'train',
    mode: 'lines',
    line: {width: 1}
  },
  validation: {
    x: [],
    y: [],
    name: 'validation',
    mode: 'lines+markers',
    line: {width: 3}
  }
};
export function plotAccuracy(batch, accuracy, set) {
  accuracyValues[set].x.push(batch);
  accuracyValues[set].y.push(accuracy);
  Plotly.newPlot(
      'accuracy-canvas',
      [accuracyValues.train, accuracyValues.validation], {
        width: 480,
        xaxis: {title: 'batch #'},
        yaxis: {
          title: 'accuracy',
          range: [0, 1]
        },
        font: {size: 18}
      });
  accuracyLabelElement.innerText =
      `last accuracy: ${(accuracy * 100).toFixed(1)}%`;
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function getModelTypeId() {
  return document.getElementById('model-type').value;
}

export function getTrainEpochs() {
  return Number.parseInt(document.getElementById('train-epochs').value);
}

export function setTrainButtonCallback(callback) {
  const trainButton = document.getElementById('train');
  const modelType = document.getElementById('model-type');
  trainButton.addEventListener('click', () => {
    trainButton.setAttribute('disabled', true);
    modelType.setAttribute('disabled', true);
    callback();
  });
}
