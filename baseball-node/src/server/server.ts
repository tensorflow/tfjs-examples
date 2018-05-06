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
import {isClassifiedPitchType} from 'baseball-pitchfx-data';
// tslint:disable-next-line:max-line-length
import {Pitch} from 'baseball-pitchfx-types';

import {PitchTypeModel} from '../pitch-type-model';

import {PitchCache} from './cache';
import {PitchPoller} from './poller';
import {Socket} from './socket';

// Enable TFJS-Node backend
bindTensorFlowBackend();

function toggleLiveData() {
  if (useLiveData) {
    poller.poller.unsubscribe();
    scheduleTrainingDataLoop();
  } else {
    clearInterval(intervalId);
    scheduleLiveDataLoop();
  }
  useLiveData = !useLiveData;
}

function scheduleTrainingDataLoop() {
  console.log('  > Using test data');
  pitchCache.loadTestData();

  intervalId = setInterval(async () => {
    await pitchModel.train(1, progress => socket.sendProgress(progress));
    socket.broadcastUpdatedPredictions();

    if (useLiveData) {
      clearInterval(intervalId);
    }
  }, 2000);
}

function scheduleLiveDataLoop() {
  console.log('  > Using live data');
  poller.poll();
  poller.poller.subscribe(async (newPitches: Pitch[]) => {
    const displayPitches =
        newPitches.filter(pitch => isClassifiedPitchType(pitch));

    pitchCache.cachePitches(displayPitches);

    if (pitchCache.trainSize() > 0) {
      console.log(`  > Training with ${pitchCache.trainSize()} live pitches`);
      await pitchModel.trainWithPitches(
          pitchCache.trainCache, progress => socket.sendProgress(progress));
    }

    if (pitchCache.queueSize() > 0) {
      socket.broadcastPredictions();
    } else {
      socket.broadcastUpdatedPredictions();
    }
  });
}

let useLiveData = false;
let intervalId: NodeJS.Timer;

const pitchModel = new PitchTypeModel();
const pitchCache = new PitchCache(pitchModel);
const poller = new PitchPoller();

const socket = new Socket(pitchCache, toggleLiveData);

async function run() {
  socket.listen();
  scheduleTrainingDataLoop();
}

run();
