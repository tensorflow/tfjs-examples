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

import'@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

tf.setBackend('tensorflow');

import {PitchTypeModel} from '../pitch-type-model';
import {sleep} from '../utils';
import {Socket} from './socket';

const TIMEOUT_BETWEEN_EPOCHS_MS = 500;

const pitchModel = new PitchTypeModel();
const socket = new Socket();

async function run() {
  socket.listen();
  socket.sendAccuracyPerClass(await pitchModel.evaluate());
  await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);

  while (true) {
    await pitchModel.train(1, progress => socket.sendProgress(progress));
    socket.sendAccuracyPerClass(
        await pitchModel.evaluate(socket.useTrainingData));
    await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);
  }
}

run();
