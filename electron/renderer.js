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
ipcRenderer.on('get-files-response', (event, arg) => {
  hideProgress();
  if (arg.foundItems.length === 0) {
    showSnackbar(
        `No match for "${arg.targetWords.join(',')}" ` +
        `after searching ${arg.numSearchedFiles} file(s). ` +
        `Elapsed time: ${arg.tElapsedMillis.toFixed(1)} ms`);
  } else {
    showSnackbar(
        `Found ${arg.foundItems.length} ` +
        `matches from ${arg.numSearchedFiles} image(s). ` +
        `Elapsed time: ${arg.tElapsedMillis.toFixed(1)} ms`);
    arg.foundItems.forEach(foundItem => {
      createFoundCard(searchResultsDiv, foundItem);
    });
  }
});

ipcRenderer.on('loading-model', (event) => {
  showProgress('Loading model...');
});

ipcRenderer.on('inference-ongoing', (event) => {
  showProgress('Classifying images...');
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

const filesDialogButton = document.getElementById('files-dialog-button');
filesDialogButton.addEventListener('click', () => {
  const targetWords = getTargetWords();
  if (targetWords == null || targetWords.length === 0) {
    showSnackbar(`You didn't specify any search words!`);
  }
  ipcRenderer.send('get-files', {targetWords});
});

const directoriesDialogButton =
    document.getElementById('directories-dialog-button');
directoriesDialogButton.addEventListener('click', () => {
  const targetWords = getTargetWords();
  if (targetWords == null || targetWords.length === 0) {
    showSnackbar(`You didn't specify any search words!`);
  }
  ipcRenderer.send('get-directories', {targetWords});
});

function limitStringToLength(str, limit = 50) {
  if (str.length <= limit) {
    return str;
  } else {
    return `...${str.slice(str.length - limit)}`;
  }
}

/**
 * Create and material-design card for a search match and add
 * it to the root div for search results.
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

  rootDiv.insertBefore(cardDiv, rootDiv.firstChild);
  cardDiv.scrollIntoView();
}

const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');

function showProgress(message) {
  progressText.textContent = message;
  progressBar.style.display = 'block';
  progressText.style.display = 'block';
}

function hideProgress(message) {
  progressBar.style.display = 'none';
  progressText.style.display = 'none';
}

hideProgress();
