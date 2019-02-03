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
import {runSeq2SeqInference} from './model';

async function init() {
  console.log(tf.version);  // DEBUG

  const model = await tf.loadModel('./model/model.json');
  model.summary();

  const inputDateString = '23DEC1996';
  // const inputDateString = '19961223';

  const getAttention = true;
  const {outputStr, attention} =
      await runSeq2SeqInference(model, inputDateString, getAttention);
  const exampleAttention = attention.squeeze([0]);
  console.log(`Ouptut date string = "${outputStr}"`);
  exampleAttention.print();
}

init();