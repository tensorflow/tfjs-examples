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
 * Addition RNN example.
 *
 * Based on tfjs example:
 *   https://github.com/tensorflow/tfjs-examples/tree/master/addition-rnn
 */

import * as tfvis from '@tensorflow/tfjs-vis';
const worker = new Worker('./worker.js');

async function runAdditionRNNDemo() {
  document.getElementById('trainModel').addEventListener('click', async () => {
    const digits = +(document.getElementById('digits')).value;
    const trainingSize = +(document.getElementById('trainingSize')).value;
    const rnnTypeSelect = document.getElementById('rnnType');
    const rnnType =
      rnnTypeSelect.options[rnnTypeSelect.selectedIndex].getAttribute(
        'value');
    const layers = +(document.getElementById('rnnLayers')).value;
    const hiddenSize = +(document.getElementById('rnnLayerSize')).value;
    const batchSize = +(document.getElementById('batchSize')).value;
    const trainIterations = +(document.getElementById('trainIterations')).value;
    const numTestExamples = +(document.getElementById('numTestExamples')).value;

    // Do some checks on the user-specified parameters.
    const status = document.getElementById('trainStatus');
    if (digits < 1 || digits > 5) {
      status.textContent = 'digits must be >= 1 and <= 5';
      return;
    }
    const trainingSizeLimit = Math.pow(Math.pow(10, digits), 2);
    if (trainingSize > trainingSizeLimit) {
      status.textContent =
        `With digits = ${digits}, you cannot have more than ` +
        `${trainingSizeLimit} examples`;
      return;
    }
    worker.postMessage({ digits, trainingSize, rnnType, layers, hiddenSize, trainIterations, batchSize, numTestExamples });
    worker.addEventListener('message', (e) => {
      if (e.data.isPredict) {
        const { i, iterations, modelFitTime, lossValues, accuracyValues } = e.data;
        document.getElementById('trainStatus').textContent =
          `Iteration ${i + 1} of ${iterations}: ` +
          `Time per iteration: ${modelFitTime.toFixed(3)} (seconds)`;
        const lossContainer = document.getElementById('lossChart');
        tfvis.render.linechart(
          lossContainer, { values: lossValues, series: ['train', 'validation'] },
          {
            width: 420,
            height: 300,
            xLabel: 'epoch',
            yLabel: 'loss',
          });

        const accuracyContainer = document.getElementById('accuracyChart');
        tfvis.render.linechart(
          accuracyContainer,
          { values: accuracyValues, series: ['train', 'validation'] }, {
            width: 420,
            height: 300,
            xLabel: 'epoch',
            yLabel: 'accuracy',
          });
      } else {
        const { isCorrect, examples } = e.data;
        const examplesDiv = document.getElementById('testExamples');
        const examplesContent = examples.map(
          (example, i) =>
            `<div class="${
            isCorrect[i] ? 'answer-correct' : 'answer-wrong'}">` +
            `${example}` +
            `</div>`);

        examplesDiv.innerHTML = examplesContent.join('\n');
      }
    });
  });
}

runAdditionRNNDemo();
