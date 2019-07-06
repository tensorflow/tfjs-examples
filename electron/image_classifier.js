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

const MOBILENET_MODEL_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'

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

  /** If the underlying model is not loaded, load it. */
  async ensureModelLoaded() {
    if (this.model == null) {
      console.log('Loading image classifier model...');

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
      foundItems.push({
        filePath: filePaths[i],
        matchWord,
        topClasses: namesAndProbs,
      });
    }
  }
  return foundItems;
}

/** Is the current environment Node.js? */
function isNode() {
  return (
      typeof process === 'object' &&
      typeof process.versions === 'object' &&
      typeof process.versions.node !== 'undefined');
}

/** Get the user's home directory (Node.js only). */
function getUserHomeDirectory() {
  // Based on:
  // https://stackoverflow.com/questions/9080085/node-js-find-home-directory-in-platform-agnostic-way
  return process.env[process.platform === 'win32' ? 'USERPROFILE' : 'HOME'];
}

