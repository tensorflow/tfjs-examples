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

import * as socketioClient from 'socket.io-client';
import Vue from 'vue';
import {AccuracyPerClass} from '../types';

const SOCKET = 'http://localhost:8001/';

// tslint:disable-next-line:no-default-export
export default Vue.extend({
  mounted: () => {
    const liveButton = document.getElementById('live-button');
    const socket = socketioClient(
        SOCKET, {reconnectionDelay: 300, reconnectionDelayMax: 300});
    socket.connect();

    socket.on('connect', () => {
      liveButton.style.display = 'block';
      liveButton.textContent = 'Test Live';
    });

    socket.on('accuracyPerClass', (accPerClass: AccuracyPerClass) => {
      plotAccuracyPerClass(accPerClass);
    });

    socket.on('disconnect', () => {
      liveButton.style.display = 'block';
      document.getElementById('waiting-msg').style.display = 'block';
      document.getElementById('table').style.display = 'none';
    });

    liveButton.onclick = () => {
      liveButton.textContent = 'Loading...';
      socket.emit('live_data', '' + true);
    };
  },
});

const BAR_WIDTH_PX = 300;

function plotAccuracyPerClass(accPerClass: AccuracyPerClass) {
  document.getElementById('table').style.display = 'block';
  document.getElementById('waiting-msg').style.display = 'none';

  const table = document.getElementById('table-rows');
  table.innerHTML = '';

  // Sort class names before displaying.
  const sortedClasses = Object.keys(accPerClass).sort();
  sortedClasses.forEach(label => {
    const scores = accPerClass[label];
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
    rowDiv.appendChild(scoreContainer);

    plotScoreBar(scores.training, scoreContainer);
    if (scores.validation) {
      document.getElementById('live-button').style.display = 'none';
      plotScoreBar(scores.validation, scoreContainer, 'validation');
    }
  });
}

function plotScoreBar(
    score: number, container: HTMLDivElement, className = '') {
  const scoreDiv = document.createElement('div');
  scoreDiv.className = 'score ' + className;
  scoreDiv.style.width = (score * BAR_WIDTH_PX) + 'px';
  scoreDiv.innerHTML = (score * 100).toFixed(1);
  container.appendChild(scoreDiv);
}
