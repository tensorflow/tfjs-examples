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

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

tf.setBackend('tensorflow');

import {Timer} from 'node-simple-timer';

import {PitchTypeModel} from '../pitch-type-model';

async function test() {
  const model = new PitchTypeModel();
  const timer = new Timer();

  for (let i = 0; i < 100; i++) {
    timer.start();
    await model.train(1, () => {});
    timer.end();
    console.log('  > epoch train time: ', timer.seconds());
    timer.reset();
  }
}

test();
