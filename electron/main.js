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
// Will be dynamically imported depending on whether the --gpu flag
// is specified.
const tf = process.argv.indexOf('--gpu') === -1 ?
    require('@tensorflow/tfjs-node') :
    require('@tensorflow/tfjs-node-gpu');

import {IMAGE_EXTENSION_NAMES, findImagesFromDirectoriesRecursive, readImageAsBase64, readImageAsTensor} from './image_utils';
import {ImageClassifier, searchForKeywords} from './image_classifier';

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

let imageClassifier;  // The ImageClassifier instance to be loaded dynamically.

/**
 * Search for images with content matching target wrods.
 *
 * @param {string[]} filePaths An array of paths to image files
 * @param {string[]} targetWords What target words to search for. An image
 *   will be considered a match if its content (as determined by
 *   `imageClassifer`) matches any of the target words.
 * @param {() => any} modelLoadingCallback An optional callback that will
 *   be invoked during loading of the model.
 * @param {() => any} inferenceCallback An optional callback that will
 *   be invoked when the model is running inference on image data.
 */
async function searchFromFiles(
    filePaths, targetWords, modelLoadingCallback, inferenceCallback) {
  if (imageClassifier == null) {
    imageClassifier = new ImageClassifier();
    if (modelLoadingCallback != null) {
      modelLoadingCallback();
    }
    await imageClassifier.ensureModelLoaded();
  }
  const {height, width} = imageClassifier.getImageSize();

  // Read the content of the image files as tensors with dimensions
  // that match the requirement of the image classifier.
  const imageTensors = [];
  for (const file of filePaths) {
    const imageTensor = await readImageAsTensor(file, height, width);
    imageTensors.push(imageTensor);
  }

  // Combine images to a batch for accelerated inference.
  const axis = 0;
  const batchImageTensor = tf.concat(imageTensors, axis);
  if (inferenceCallback != null) {
    inferenceCallback();
  }

  // Run inference.
  const t0 = tf.util.now();
  const classNamesAndProbs = await imageClassifier.classify(batchImageTensor);
  const tElapsedMillis = tf.util.now() - t0;

  const foundItems = searchForKeywords(
      classNamesAndProbs, filePaths, targetWords);
  for (const foundItem of foundItems) {
    try {
      foundItem.imageBase64 = await readImageAsBase64(foundItem.filePath);
    } catch (err) {
      // Guards against `readImageAsBase64` failures.
    }
  }

  // TensorFlow.js memory cleanup.
  tf.dispose([imageTensors, batchImageTensor, imageTensors]);

  return {
    targetWords,
    numSearchedFiles: filePaths.length,
    foundItems,
    tElapsedMillis
  };
}

/**
 * Read a number of image files as a structured, batched number array.
 *
 * @param {string[]} imageFilePaths Paths to the image files to read from.
 * @param {number} height Image height, in pixels.
 * @param {number} width Image width, in pixels.
 * @return {number[][][][][]} A nested number array of effective shape
 *   `[numImages, height, width, channels]`.
 */
async function readImagesAsBatchedStructuredArray(
    imageFilePaths, height, width) {
  const imageTensors = [];
  for (const filePath of imageFilePaths) {
    imageTensors.push(await readImageAsTensor(filePath, height, width));
  }
  const axis = 0;
  const imageTensorData = await tf.concat(imageTensors, axis).array();
  tf.dispose(imageTensors);
  return imageTensorData;
}

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
      event.sender.send('reading-images');
      // Perform inference using frontend model.
      // Read images and send them to the frontend via IPC.
      const imageTensorData = await readImagesAsBatchedStructuredArray(
          imageFilePaths, arg.imageHeight, arg.imageWidth);
      console.log(
          `Sending data from ${imageTensorData.length} image(s) to frontend ` +
          `for inferene`);
      event.sender.send('frontend-inference-data', {
        imageFilePaths,
        imageTensorData
      });
    } else {
      // Perform inference in the backend (i.e., in this process).
      const results = await searchFromFiles(
          imageFilePaths, arg.targetWords,
          () => event.sender.send('loading-model'),
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
      event.sender.send('reading-images');
      // Perform inference using frontend model.
      // Read images and send them to the frontend via IPC.
      const imageTensorData = await readImagesAsBatchedStructuredArray(
          imageFilePaths, arg.imageHeight, arg.imageWidth);
      console.log(
          `Sending data from ${imageTensorData.length} image(s) to frontend ` +
          `for inferene`);
      event.sender.send('frontend-inference-data', {
        imageFilePaths,
        imageTensorData
      });
    } else {
      // Perform inference in the backend (i.e., in this process).
      const results = await searchFromFiles(
          imageFilePaths, arg.targetWords,
          () => event.sender.send('loading-model'),
          () => event.sender.send('inference-ongoing'));
      event.sender.send('search-response', results);
    }
  });
});
