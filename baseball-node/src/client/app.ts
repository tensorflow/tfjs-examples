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

// tslint:disable-next-line:max-line-length
import {PitchPredictionMessage, PitchPredictionUpdateMessage} from 'baseball-pitchfx-types';
import {UniqueQueue} from 'containers.js';
import * as socketioClient from 'socket.io-client';
import embed from 'vega-embed';
import Vue from 'vue';

import {TrainProgress} from '../abstract-pitch-model';
import {AccuracyPerClass} from '../types';

import Pitch from './Pitch.vue';

const maxPitches = 6 * 3;  // 3 each row.

const SOCKET = 'http://localhost:8001/';

const data = {
  predictionsQueue: new UniqueQueue<PitchPredictionMessage>(maxPitches),
  predictions: [] as PitchPredictionMessage[],
  predictionMap: new Map<string, number>()
};

const accuracy: Array<{batch: number, accuracy: number}> = [];

// tslint:disable-next-line:no-default-export
export default Vue.extend({
  data() {
    return data;
  },
  components: {Pitch},
  // tslint:disable-next-line:object-literal-shorthand
  mounted: function() {
    const socket = socketioClient(
        SOCKET, {reconnectionDelay: 300, reconnectionDelayMax: 300});
    socket.connect();

    socket.on('pitch_predictions', (data: PitchPredictionMessage[]) => {
      data.forEach((prediction) => {
        this.predictionsQueue.push(prediction);
      });
      this.predictions = this.predictionsQueue.values();
      this.predictionMap.clear();
      for (let i = 0; i < this.predictions.length; i++) {
        this.predictionMap.set(this.predictions[i].uuid, i);
      }
    });

    socket.on('accuracyPerClass', (accPerClass: AccuracyPerClass) => {
      plotAccuracyPerClass(accPerClass);
    });

    socket.on('prediction_updates', (data: PitchPredictionUpdateMessage[]) => {
      data.forEach((update) => {
        const index = this.predictionMap.get(update.uuid);
        if (index !== undefined) {
          this.predictions[index].pitch_classes = update.pitch_classes;
          this.predictions[index].strike_zone_classes =
              update.strike_zone_classes;
        }
      });
    });

    socket.on('progress', (progress: TrainProgress) => plotProgress(progress));

    socket.on('disconnect', () => {
      this.predictionMap.clear();
      this.predictions = [];
      this.predictionsQueue.clear();
    });
  }
});

const MAX_NUM_POINTS = 100;

function subsample(accuracy: Array<{batch: number, accuracy: number}>) {
  const skip = Math.max(1, accuracy.length / MAX_NUM_POINTS);
  const result: Array<{batch: number, accuracy: number}> = [];
  for (let i = 0; i < accuracy.length; i += skip) {
    result.push(accuracy[Math.round(i)]);
  }
  return result;
}

function plotAccuracyPerClass(accPerClass: AccuracyPerClass) {
  const table = document.getElementById('table');
  table.innerHTML = '';

  const BAR_WIDTH_PX = 300;

  for (const label in accPerClass) {
    // Row.
    const rowDiv = document.createElement('div');
    rowDiv.className = 'row';
    table.appendChild(rowDiv);

    // Label.
    const labelDiv = document.createElement('div');
    labelDiv.innerText = label;
    labelDiv.className = 'label';
    rowDiv.appendChild(labelDiv);

    // Score.
    const scoreContainer = document.createElement('div');
    scoreContainer.className = 'score-container';
    scoreContainer.style.width = BAR_WIDTH_PX + 'px';

    const scoreDiv = document.createElement('div');
    scoreDiv.className = 'score';
    const score = accPerClass[label].training;
    scoreDiv.style.width = (score * BAR_WIDTH_PX) + 'px';
    scoreDiv.innerHTML = (score * 100).toFixed(1) + '%';

    scoreContainer.appendChild(scoreDiv);
    rowDiv.appendChild(scoreContainer);
  }
}

function plotProgress(progress: TrainProgress) {
  accuracy.push({batch: accuracy.length + 1, accuracy: progress.accuracy});
  embed(
      '#accuracyCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': subsample(accuracy)},
        'width': 260,
        'mark': {'type': 'line', 'legend': null, 'orient': 'vertical'},
        'encoding': {
          'x': {'field': 'batch', 'type': 'quantitative'},
          'y': {'field': 'accuracy', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal', 'legend': null},
        },
      },
      {'width': 360});
}
