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

import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from './imagenet_classes';
import {readImageAsTensor} from './image_utils';

const MOBILENET_MODEL_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'

/**
 * A classifier for images.
 *
 * It uses an underlying TensorFlow.js convolutional neural network
 * to label a batch of input images. The labels are from the ImageNet
 * dataset and can be seen in `./imagenet_classes.js`.
 */
export class ImageClassifier {
  constructor() {
    this.model = null;
  }

  /**
   * Perform classification on a batch of image tensors.
   *
   * @param {tf.Tensor} images Batch image tensor of shape
   *   `[numExamples, height, width, channels]`. The values of `heigth`,
   *   `width` and `channel` must match the underlying MobileNetV2 model
   *   (default: 224, 224, 3).
   * @param {number} topK How many results with top probability / logit values
   *   to return for each example.
   * @return {Array<{className: string, prob: number}>} An array of classes
   *   with the highest `topK` probability scores, sorted in the descending
   *   order of the probability scores. Each element of the array corresponds
   *   to one example in `images`. The order of the elements matches that
   *   of `images`.
   */
  async classify(images, topK = 5) {
    await this.ensureModelLoaded();
    return tf.tidy(() => {
      const probs = this.model.predict(images);
      const sorted = true;
      const {values, indices} = tf.topk(probs, topK, sorted);

      const classProbs = values.arraySync();
      const classIndices = indices.arraySync();

      const results = [];
      classIndices.forEach((indices, i) => {
        const classesAndProbs = [];
        indices.forEach((index, j) => {
          classesAndProbs.push({
            className: IMAGENET_CLASSES[index],
            prob: classProbs[i][j]
          });
        });
        results.push(classesAndProbs);
      })

      return results;
    });
  }

  /**
   * If the underlying model is not loaded, load it.
   *
   * @param {() => any} loadingCallback An optional callback function that will
   *   be invoked when the model is being loaded.
   */
  async ensureModelLoaded(loadingCallback) {
    if (this.model == null) {
      console.log('Loading image classifier model...');
      if (loadingCallback != null) {
        loadingCallback();
      }

      let cachedModelJsonUrl;
      if (isNode()) {
        // Attempt to find and load model cached on file system if running
        // in Node.js.
        const fs = require('fs');
        const path = require('path');
        const cachedModelJsonPath = path.join(
            this.getFileSystemCacheDirectory_(), 'model.json');
        if (fs.existsSync(cachedModelJsonPath)) {
          cachedModelJsonUrl = `file://${cachedModelJsonPath}`;
          console.log(`Found cached model at ${cachedModelJsonUrl}`);
        }
      }

      console.time('Model loading');
      this.model = await tf.loadLayersModel(
          cachedModelJsonUrl == null ?
          MOBILENET_MODEL_URL : cachedModelJsonUrl);
      console.timeEnd('Model loading');

      if (isNode() && cachedModelJsonUrl == null) {
        // Cache model on file system if running in Node.js.
        const cacheDir = this.getFileSystemCacheDirectory_();
        try {
          await this.model.save(`file://${cacheDir}`);
          console.log(`Cached model at ${cacheDir}`);
        } catch (err) {
          console.warn(`Failed to save model at cache directory: ${cacheDir}`);
        }
      }
    }
  }

  /**
   * Search for images with content matching target wrods.
   *
   * @param {string[]} filePaths An array of paths to image files
   * @param {string[]} targetWords What target words to search for. An image
   *   will be considered a match if its content (as determined by
   *   `imageClassifer`) matches any of the target words.
   * @param {() => any} inferenceCallback An optional callback that will
   *   be invoked when the model is running inference on image data.
   */
  async searchFromFiles(filePaths, targetWords, inferenceCallback) {
    // Read the content of the image files as tensors with dimensions
    // that match the requirement of the image classifier.
    const {height, width} = this.getImageSize();
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
    const classNamesAndProbs = await this.classify(batchImageTensor);
    const tElapsedMillis = tf.util.now() - t0;

    const foundItems = searchForKeywords(
        classNamesAndProbs, filePaths, targetWords);

    // TensorFlow.js memory cleanup.
    tf.dispose([imageTensors, batchImageTensor, imageTensors]);

    return {
      targetWords,
      numSearchedFiles: filePaths.length,
      foundItems,
      tElapsedMillis
    };
  }

  /** Get the required image sizes (height and width). */
  getImageSize() {
    if (this.model == null) {
      throw new Error(
          `Model is not loaded yet. Call ensureModelLoaded() first.`);
    }
    return {
      height: this.model.inputs[0].shape[1],
      width: this.model.inputs[0].shape[2]
    }
  }

  getFileSystemCacheDirectory_() {
    const path = require('path');
    return path.join(getUserHomeDirectory(), '.tfjs-examples-electron');
  }
}

/**
 * Search for target words in an array of class names and corresponding
 * probabilities.
 *
 * This search is necessary because the class names output by the
 * TensorFlow.js model are not isolated English words, instead they long
 * phrases such as "tiger shark, Galeocerdo cuvieri". We need to break
 * these labels into words and match them against the target words
 * provided by the app's user (e.g., "shark").
 *
 * @param {Array<{className: string, prob: number}>} classNamesAndProbs
 *   An array of `N` classification results, each of which is an object
 *   mapping a class name (`className`) to a probability score (`prob`).
 * @param {string[]} The file paths of the image files. Must have the
 *   same length as `classNamesAndProbs`.
 * @param {string[]} targetWords An array of target words to search for
 *   in the results.
 * @returns {Array<{filePath: string, matchWord: string, topClasses: string}>}
 *   All matches to the target words.
 */
export function searchForKeywords(classNamesAndProbs, filePaths, targetWords) {
   // Filter through the output class names and probilities to look for
  // matches.
  const foundItems = [];
  for (let i = 0; i < classNamesAndProbs.length; ++i) {
    const namesAndProbs = classNamesAndProbs[i];
    let matchWord = null;
    for (const nameAndProb of namesAndProbs) {
      const classTokens = nameAndProb.className.toLowerCase().trim()
          .replace(/[,\/]/g, ' ')
          .split(' ').filter(x => x.length > 0);
      for (const word of targetWords) {
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
      foundItems.push({
        filePath: filePaths[i],
        matchWord,
        topClasses: namesAndProbs,
      });
    }
  }
  return foundItems;
}

/**
 * Is the current environment Node.js?
 *
 * This logic is specific to Electron, because it checks
 * `process.type`.
 */
function isNode() {
  return (
      typeof process === 'object' &&
      typeof process.versions === 'object' &&
      typeof process.versions.node !== 'undefined' &&
      process.type !== 'renderer');
}

/** Get the user's home directory (Node.js only). */
function getUserHomeDirectory() {
  // Based on:
  // https://stackoverflow.com/questions/9080085/node-js-find-home-directory-in-platform-agnostic-way
  return process.env[process.platform === 'win32' ? 'USERPROFILE' : 'HOME'];
}

