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

const MOBILENET_MODEL_TFHUB_URL =
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2'

export class ImageClassifier {
  constructor() {
    this.model = null;
  }

  /**
   * TODO(cais): Doc string.
   * @param {*} images
   * @param {*} topK
   * @return
   */
  async classify(images, topK = 5) {
    images.min().print();  // DEBUG
    images.max().print();  // DEBUG
    await this.ensureModelLoaded();

    console.log('Calling predict()');  // DEBUG
    const probs = this.model.predict(images);
    console.log('DONE calling predict()');  // DEBUG
    probs.print();  // DEBUG
    const sorted = true;
    const {values, indices} = tf.topk(probs, topK, sorted);
    values.print();  // DEBUG
    indices.print();  // DEBUG

    const classProbs = await values.array();
    const classIndices = await indices.array();

    const results = [];
    classIndices.forEach((indices, i) => {
      const classesAndProbs = [];
      indices.forEach((index, j) => {
        classesAndProbs.push({
          className: IMAGENET_CLASSES[index - 1],
          prob: classProbs[i][j]
        });
      });
      results.push(classesAndProbs);
    })

    console.log(results);  // DEBUG
    return results;
  }

  async ensureModelLoaded() {
    if (this.model == null) {
      console.log('Loading image classifier model...');
      this.model =
          await tf.loadGraphModel(MOBILENET_MODEL_TFHUB_URL, {fromTFHub: true});
    }
  }

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

}