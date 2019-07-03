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

import {MDCSnackbar} from '@material/snackbar';
import {ipcRenderer} from 'electron';

const searchResultsDiv = document.getElementById('search-results');
ipcRenderer.on('get-files-response', (event, found) => {
  if (found.length === 0) {

  } else {
    found.forEach(foundItem => {
      createFoundCard(searchResultsDiv, foundItem);
    });
  }
});

const targetWordsInput = document.getElementById('target-words');
function getTargetWords() {
  return targetWordsInput.value.trim().split(',')
      .filter(x => x.length > 0).map(x => x.trim().toLowerCase());
}

const snackbar = new MDCSnackbar(document.getElementById('main-snackbar'));
function showSnackbar(message, timeoutMs = 4000) {
  snackbar.labelText = message;
  snackbar.timeoutMs = timeoutMs;
  snackbar.open();
}

showSnackbar('ready!!');  // DEBUG

const filesDialogButton = document.getElementById('files-dialog-button');
filesDialogButton.addEventListener('click', () => {
  const targetWords = getTargetWords();
  if (targetWords == null || targetWords.length === 0) {
    // TODO(cais):
    return;
  }
  ipcRenderer.send('get-files', {targetWords});
});

// TODO(cais):
// const directoriesDialogButton =
//     document.getElementById('directories-dialog-button');
// directoriesDialogButton.addEventListener('click', () => {
//   ipcRenderer.send('get-directories');
// });

function limitStringToLength(str, limit = 50) {
  if (str.length <= limit) {
    return str;
  } else {
    return `...${str.slice(limit + 3)}`;
  }
}

/**
 * Create and material-design card for a search match.
 */
function createFoundCard(rootDiv, foundItem) {
  const cardDiv = document.createElement('div');
  cardDiv.classList.add('mdl-card');
  cardDiv.classList.add('mdl-shadow--2dp');
  cardDiv.classList.add('search-result');

  const titleDiv = document.createElement('div');
  titleDiv.classList.add('mdl-card__title');
  titleDiv.textContent = foundItem.matchWord;
  cardDiv.appendChild(titleDiv);

  const imgDiv = document.createElement('img');
  imgDiv.classList.add('search-result-thumbnail')
  imgDiv.src = foundItem.imageBase64;
  cardDiv.appendChild(imgDiv);

  const pathDiv = document.createElement('div');
  pathDiv.classList.add('mdl-card--border');
  pathDiv.classList.add('search-result-file-path');
  pathDiv.textContent = limitStringToLength(foundItem.filePath);
  cardDiv.appendChild(pathDiv);

  const topKDiv = document.createElement('div');
  topKDiv.classList.add('mdl-card--border');
  topKDiv.classList.add('search-result-top-k');
  const ul = document.createElement('ul');
  for (const classNameAndProb of foundItem.topClasses) {
    const li = document.createElement('li');
    li.textContent =
        `${classNameAndProb.className}: ${classNameAndProb.prob.toFixed(2)}`;
    ul.appendChild(li);
  }
  topKDiv.appendChild(ul);
  cardDiv.appendChild(topKDiv);

  rootDiv.appendChild(cardDiv);
  cardDiv.scrollIntoView();
}