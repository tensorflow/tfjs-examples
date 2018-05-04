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

import {bindTensorFlowBackend} from '@tensorflow/tfjs-node';

import {PitchTypeModel} from '../pitch-type-model';

import {Socket} from './socket';

bindTensorFlowBackend();

const pitchModel = new PitchTypeModel();
const socket = new Socket(pitchModel);

async function run() {
  socket.listen();
  await pitchModel.train(1);

  setInterval(async () => {
    await pitchModel.train(1);
    socket.broadcastUpdatedPredictions();

    // TODO(kreeger): Showcase live data.

    // const rand = (Math.floor(Math.random() * 5) + 2);
    // if (count % rand === 0) {
    //   // socket.addNewRandom(1);
    // } else {
    //   socket.broadcastUpdatedPredictions();
    // }
  }, 5000);
}

run();
