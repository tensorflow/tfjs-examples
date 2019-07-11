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
import * as tf from '@tensorflow/tfjs';
import {ipcRenderer} from 'electron';

import {ImageClassifier} from './image_classifier';

const searchResultsDiv = document.getElementById('search-results');

/**
 * IPC handle for classification results.
 *
 * `arg` is expected to contain the following fields:
 *   - targetWords: What words were used for the search.
 *   - numSearchedFiles: How many files have been searched over.
 *   - tElapsedMillis: How long the inference part of the search took
 *     (in milliseconds).
 *   - foundItems: Image files with classification results that match
 *     any of the target words.
 */
ipcRenderer.on('search-response', (event, arg) => {
  displaySearchResults(arg);
});

/** IPC handler for the backend's "Reading images" status. */
ipcRenderer.on('reading-images', (event) => {
  showProgress('Reading images...');
});

/** IPC handler for the "model is be loaded" event. */
ipcRenderer.on('loading-model', (event) => {
  showProgress('Loading model...');
});

/** IPC handler for the "model is running inference" event. */
ipcRenderer.on('inference-ongoing', (event) => {
  showProgress('Classifying images...');
});

/**
 * IPC handler for image data read from the backend process.
 *
 * The image classifier model will be used to perform inference
 * on the images, and the matches (if any) will be displayed on
 * screen.
 */
ipcRenderer.on('frontend-inference-data', async (event, arg) => {
  showProgress('Classifying images in frontend...');
  await imageClassifer.ensureModelLoaded(
      () => showProgress('Loading frontend model...'));
  const results = await imageClassifer.searchFromFiles(
      arg.imageFilePaths, getTargetWords(),
      () => showProgress('Running image search in frontend...'));
  displaySearchResults(results);
});

const imageClassifer = new ImageClassifier();

/** Parse the target words for search from the text box. */
const targetWordsInput = document.getElementById('target-words');
function getTargetWords() {
  return targetWordsInput.value.trim().split(',')
      .filter(x => x.length > 0).map(x => x.trim().toLowerCase());
}

const snackbar = new MDCSnackbar(document.getElementById('main-snackbar'));

/**
 * Display result results (from backend or frontend).
 *
 * @param {object} results Search result object. Assumed to have the following
 *   fields:
 *   - targetWords {string[]} The target words searched for.
 *   - numSearchedFiles {number} Total number of image files searched over.
 *   - foundItems {Array} An array of found items (i.e., images with top-5)
 *     classification results matching any of the elements of `targetWords`.
 *   - tElapsedMillis {number} The amount of time (in millis) spent on model
 *     inference.
 */
function displaySearchResults(results) {
  hideProgress();
  if (results.foundItems.length === 0) {
    showSnackbar(
        `No match for "${results.targetWords.join(',')}" ` +
        `after searching ${results.numSearchedFiles} file(s). ` +
        `Model inference took ${results.tElapsedMillis.toFixed(1)} ms`);
  } else {
    showSnackbar(
        `Found ${results.foundItems.length} ` +
        `matches from ${results.numSearchedFiles} image(s). ` +
        `Model inference took ${results.tElapsedMillis.toFixed(1)} ms`);
    results.foundItems.forEach(foundItem => {
      createFoundCard(searchResultsDiv, foundItem);
    });
  }
}

/**
 * Display a snackbar message on the screen.
 *
 * @param {string} message The message to be displayed.
 * @param {number} timeoutMillis How many millliseconds the message
 *   will stay on the screen before disappearing.
 */
function showSnackbar(message, timeoutMillis = 4000) {
  snackbar.labelText = message;
  snackbar.timeoutMs = timeoutMillis;
  snackbar.open();
}

const filesDialogButton = document.getElementById('files-dialog-button');
const frontendInferenceCheckbox =
    document.getElementById('frontend-inference-checkbox');

/** The callback for selecting a number of files to search over. */
filesDialogButton.addEventListener('click', async () => {
  const targetWords = getTargetWords();
  if (targetWords == null || targetWords.length === 0) {
    showSnackbar(`You didn't specify any search words!`);
  }
  const frontendInference = frontendInferenceCheckbox.checked;
  if (frontendInference) {
    await imageClassifer.ensureModelLoaded(
        () => showProgress('Loading frontend model...'));
  }
  ipcRenderer.send('get-files', {targetWords, frontendInference});
});

const directoriesDialogButton =
    document.getElementById('directories-dialog-button');

/** The callback for selecting a number of folder to search in, recursively. */
directoriesDialogButton.addEventListener('click', async () => {
  const targetWords = getTargetWords();
  if (targetWords == null || targetWords.length === 0) {
    showSnackbar(`You didn't specify any search words!`);
  }
  const frontendInference = frontendInferenceCheckbox.checked;
  if (frontendInference) {
    await imageClassifer.ensureModelLoaded(
        () => showProgress('Loading frontend model...'));
  }
  ipcRenderer.send('get-directories', {targetWords, frontendInference});
});

/** Helper method for limiting the number of characters shown on screen. */
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
  imgDiv.classList.add('search-result-thumbnail');
  imgDiv.src = `file://${foundItem.filePath}`;
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
    if (classNameAndProb.prob >= 0.001) {
      li.textContent =
          `${classNameAndProb.className}: ${classNameAndProb.prob.toFixed(3)}`;
      ul.appendChild(li);
    }
  }
  topKDiv.appendChild(ul);
  cardDiv.appendChild(topKDiv);

  rootDiv.insertBefore(cardDiv, rootDiv.firstChild);
  cardDiv.scrollIntoView();

  updateClearSearchResultsButtonStatus();
}

const clearSearchResultsButton =
    document.getElementById('clear-search-results');
function updateClearSearchResultsButtonStatus() {
  clearSearchResultsButton.style.display =
      searchResultsDiv.firstChild ? 'block' : 'none';
}
updateClearSearchResultsButtonStatus();

clearSearchResultsButton.addEventListener('click', () => {
  while(searchResultsDiv.firstChild) {
    searchResultsDiv.removeChild(searchResultsDiv.firstChild);
  }
  updateClearSearchResultsButtonStatus();
});

const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');

/** Display an indeterminate progress bar with message. */
function showProgress(message) {
  progressText.textContent = message;
  progressBar.style.display = 'block';
  progressText.style.display = 'block';
}

/** Display the indeterminate progress bar. */
function hideProgress(message) {
  progressBar.style.display = 'none';
  progressText.style.display = 'none';
}

hideProgress();
