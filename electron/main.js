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
    height: 660,
    width: 1000,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
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

let imageClassifier;

ipcMain.on('get-files', (event, arg) => {
  console.log('get-files: targetWords:', arg.targetWords);  // DEBUG

  dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [{
      name: 'Images',
      extensions: ['jpg', 'jpeg', 'png']
    }]
  }, async (files) => {
    console.log(`files:`, files);  // DEBUG
    if (files == null || files.length === 0) {
      // TODO(cais): This should send an IPC to the renderer and show a
      // snackbar.
      dialog.showErrorBox(`You didn't select any files!`);
      return;
    }

    if (imageClassifier == null) {
      imageClassifier = new ImageClassifier();
      await imageClassifier.ensureModelLoaded();
    }
    const {height, width} = imageClassifier.getImageSize();

    const imageTensors = [];
    for (const file of files) {
      const imageTensor = await readImageAsTensor(file, height, width);
      imageTensors.push(imageTensor);
    }

    const axis = 0;
    const batchImageTensor = tf.concat(imageTensors, axis);
    const classNamesAndProbs = await imageClassifier.classify(batchImageTensor);
    console.log(classNamesAndProbs);  // DEBUG

    // Filter through the output class names and probilities.
    const found = [];
    for (let i = 0; i < classNamesAndProbs.length; ++i) {
      const namesAndProbs = classNamesAndProbs[i];
      let matchWord = null;
      for (const nameAndProb of namesAndProbs) {
        for (const word of arg.targetWords) {
          const classTokens = nameAndProb.className.toLowerCase().trim()
              .replace(/[,\/]/g, ' ')
              .split(' ').filter(x => x.length > 0);
          console.log(classTokens);  // DEBUG
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
        found.push(Object.assign({
          filePath: files[i],
          matchWord,
          imageBase64: await readImageAsBase64(files[i]),
          topClasses: namesAndProbs
        }))
      }
    }

    tf.dispose([imageTensors, batchImageTensor, imageTensors]);
    event.sender.send('get-files-response', found);
  });
});

ipcMain.on('get-directories', (event, arg) => {
  // TODO(cais): Implement.
});
