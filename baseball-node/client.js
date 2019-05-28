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

import io from 'socket.io-client';
const evalTestButton = document.getElementById('eval-test-button');

const socket =
    io('http://localhost:8001',
       {reconnectionDelay: 300, reconnectionDelayMax: 300});

const BAR_WIDTH_PX = 300;

evalTestButton.onclick = () => {
  evalTestButton.textContent = 'Loading...';
  socket.emit('test_data', 'true');
};

socket.on('connect', () => {
  evalTestButton.style.display = 'block';
  evalTestButton.textContent = 'Eval Test';
});

socket.on('accuracyPerClass', (accPerClass) => {
  plotAccuracyPerClass(accPerClass);
});

socket.on('disconnect', () => {
  evalTestButton.style.display = 'block';
  document.getElementById('waiting-msg').style.display = 'block';
  document.getElementById('table').style.display = 'none';
});

function plotAccuracyPerClass(accPerClass) {
  console.log(accPerClass);
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
      document.getElementById('eval-test-button').style.display = 'none';
      plotScoreBar(scores.validation, scoreContainer, 'validation');
    }
  });
}

function plotScoreBar(score, container, className = '') {
  const scoreDiv = document.createElement('div');
  scoreDiv.className = 'score ' + className;
  scoreDiv.style.width = (score * BAR_WIDTH_PX) + 'px';
  scoreDiv.innerHTML = (score * 100).toFixed(1);
  container.appendChild(scoreDiv);
}
