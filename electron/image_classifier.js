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
            className: IMAGENET_CLASSES[index - 1],
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
      this.model =
          await tf.loadGraphModel(MOBILENET_MODEL_TFHUB_URL, {fromTFHub: true});
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

}
