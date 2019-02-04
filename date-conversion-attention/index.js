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
 * Addition RNN example.
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {INPUT_LENGTH} from './date_format';
import {runSeq2SeqInference} from './model';

const inputDateString = document.getElementById('input-date-string');
const outputDateString = document.getElementById('output-date-string');
const attentionHeatmap = document.getElementById('attention-heatmap');

let model;

inputDateString.addEventListener('input', async () => {
  let inputStr = inputDateString.value.trim().toUpperCase();
  if (inputStr.length < 6) {
    outputDateString.value = '';
    return;
  }

  if (inputStr.length > INPUT_LENGTH) {
    inputStr = inputStr.slice(0, INPUT_LENGTH);
  }

  try {
    const getAttention = true;
    const {outputStr, attention} =
        await runSeq2SeqInference(model, inputStr, getAttention);
    outputDateString.value = outputStr;

    console.log(attentionHeatmap);  // DEBUG
    console.log(attention.shape);
    const xLabels = outputStr.split('').map((char, i) => `(${i + 1}) "${char}"`);
    const yLabels = [];
    for (let i = 0; i < INPUT_LENGTH; ++i) {
      if (i < inputStr.length) {
        yLabels.push(`${i + 1} "${inputStr[i]}"`);
      } else {
        yLabels.push(`${i + 1} ""`);
      }
    }
    await tfvis.render.heatmap({
      values: attention.squeeze([0]),
      xLabels,
      yLabels
    }, attentionHeatmap, {
      width: 600,
      height: 360,
      xLabel: 'Output characters',
      yLabel: 'Input characters'
    });
  } catch (err) {
    outputDateString.value = err.message;
    console.error(err);
  }
  // exampleAttention.print();
});

async function init() {
  outputDateString.value = 'Loading model...';

  model = await tf.loadModel('./model/model.json');
  model.summary();

  inputDateString.dispatchEvent(new Event('input'));
}

init();
