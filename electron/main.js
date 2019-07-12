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

import {app, dialog, ipcMain, BrowserWindow} from 'electron';
import '@tensorflow/tfjs-node';

import {IMAGE_EXTENSION_NAMES, findImagesFromDirectoriesRecursive} from './image_utils';
import {ImageClassifier} from './image_classifier';

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    height: 800,
    width: 1200
  });

  mainWindow.loadFile('index.html');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', () => {
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

const imageClassifier = new ImageClassifier();

/** IPC handle for searching over files. */
ipcMain.on('get-files', (event, arg) => {
  dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [{
      name: 'Images',
      extensions: IMAGE_EXTENSION_NAMES
    }]
  }, async (imageFilePaths) => {
    if (imageFilePaths == null ||
        Array.isArray(imageFilePaths) && imageFilePaths.length === 0) {
      // Handle cases in which no file is selected.
      return;
    }
    if (arg.frontendInference) {
      event.sender.send('frontend-inference-data', {imageFilePaths});
    } else {
      await imageClassifier.ensureModelLoaded(
          () => event.sender.send('loading-model'));
      // Perform inference in the backend (i.e., in this process).
      const results = await imageClassifier.searchFromFiles(
          imageFilePaths, arg.targetWords,
          () => event.sender.send('inference-ongoing'));
      event.sender.send('search-response', results);
    }
  });
});

/** IPC handle for searching in directories, recursively. */
ipcMain.on('get-directories', (event, arg) => {
  dialog.showOpenDialog({
    properties: ['openDirectory', 'multiSelections']
  }, async (dirPaths) => {
    if (dirPaths == null || Array.isArray(dirPaths) && dirPaths.length === 0) {
      // Handle cases in which no directory is selected.
      return;
    }
    const imageFilePaths = [];
    for (const dirPath of dirPaths) {
      imageFilePaths.push(...findImagesFromDirectoriesRecursive(dirPath));
    }
    if (imageFilePaths.length === 0) {
      // TODO(cais): in case no image exists in the selected directories (in a
      // recursive fashion), use IPC to show a snackbar in the frontend.
      return;
    }
    if (arg.frontendInference) {
      event.sender.send('frontend-inference-data', {imageFilePaths});
    } else {
      await imageClassifier.ensureModelLoaded(
          () => event.sender.send('loading-model'));
      // Perform inference in the backend (i.e., in this process).
      const results = await imageClassifier.searchFromFiles(
          imageFilePaths, arg.targetWords,
          () => event.sender.send('inference-ongoing'));
      event.sender.send('search-response', results);
    }
  });
});
