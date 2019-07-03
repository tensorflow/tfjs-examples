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

import * as fs from 'fs';
import * as path from 'path';

import {app, dialog, ipcMain, BrowserWindow} from 'electron';
// Will be dynamically imported depending on whether the --gpu flag
// is specified.
const tf = process.argv.indexOf('--gpu') === -1 ?
    require('@tensorflow/tfjs-node') :
    require('@tensorflow/tfjs-node-gpu');

import {readImageAsBase64, readImageAsTensor} from './image_utils';
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
  if (process.platfor !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

const IMAGE_EXTENSION_NAMES = ['jpg', 'jpeg', 'png'];

function findImagesFromDirectoriesRecursive(dirPath) {
  const imageFilePaths = [];
  const items = fs.readdirSync(dirPath);
  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    if (fs.lstatSync(fullPath).isDirectory()) {
      try {
        imageFilePaths.push(...findImagesFromDirectoriesRecursive(fullPath));
      } catch (err) {}
    } else {
      let extMatch = false;
      for (const extName of IMAGE_EXTENSION_NAMES) {
        if (item.toLowerCase().endsWith(extName)) {
          extMatch = true;
          break;
        }
      }
      if (extMatch) {
        imageFilePaths.push(fullPath);
      }
    }
  }
  return imageFilePaths;
}

let imageClassifier;

async function searchFromFiles(filePaths, targetWords) {
  if (imageClassifier == null) {
    imageClassifier = new ImageClassifier();
    await imageClassifier.ensureModelLoaded();
  }
  const {height, width} = imageClassifier.getImageSize();

  const imageTensors = [];
  for (const file of filePaths) {
    const imageTensor = await readImageAsTensor(file, height, width);
    imageTensors.push(imageTensor);
  }

  const axis = 0;
  const batchImageTensor = tf.concat(imageTensors, axis);
  const classNamesAndProbs = await imageClassifier.classify(batchImageTensor);
  console.log(classNamesAndProbs);  // DEBUG

  // Filter through the output class names and probilities.
  const foundItems = [];
  for (let i = 0; i < classNamesAndProbs.length; ++i) {
    const namesAndProbs = classNamesAndProbs[i];
    let matchWord = null;
    for (const nameAndProb of namesAndProbs) {
      for (const word of targetWords) {
        const classTokens = nameAndProb.className.toLowerCase().trim()
            .replace(/[,\/]/g, ' ')
            .split(' ').filter(x => x.length > 0);
        if (classTokens.indexOf(word) !== -1) {
          matchWord = word;
          break;
        }
      }
      if (matchWord != null) {
        break;
      }
    }
    if (matchWord != null) {
      foundItems.push(Object.assign({
        filePath: filePaths[i],
        matchWord,
        imageBase64: await readImageAsBase64(filePaths[i]),
        topClasses: namesAndProbs
      }))
    }
  }

  tf.dispose([imageTensors, batchImageTensor, imageTensors]);
  return {
    targetWords,
    numSearchedFiles: filePaths.length,
    foundItems
  };
}

ipcMain.on('get-files', (event, arg) => {
  dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [{
      name: 'Images',
      extensions: IMAGE_EXTENSION_NAMES
    }]
  }, async (filePaths) => {
    if (filePaths == null || filePaths.length === 0) {
      // TODO(cais): This should send an IPC to the renderer and show a
      // snackbar.
      dialog.showErrorBox(`You didn't select any files!`);
      return;
    }
    const results = await searchFromFiles(filePaths, arg.targetWords);
    event.sender.send('get-files-response', results);
  });
});

ipcMain.on('get-directories', (event, arg) => {
  dialog.showOpenDialog({
    properties: ['openDirectory', 'multiSelections']
  }, async (dirPaths) => {
    const imageFilePaths = [];
    for (const dirPath of dirPaths) {
      imageFilePaths.push(...findImagesFromDirectoriesRecursive(dirPath));
    }

    const results = await searchFromFiles(imageFilePaths, arg.targetWords);
    event.sender.send('get-files-response', results);
  });
});
