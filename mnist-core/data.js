/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tf from 'deeplearn';

const TRAIN_TEST_RATIO = 5 / 6;

const mnistConfig = {
  'data': [
    {
      'name': 'images',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_images.png',
      'dataType': 'png',
      'shape': [28, 28, 1]
    },
    {
      'name': 'labels',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_labels_uint8',
      'dataType': 'uint8',
      'shape': [10]
    }
  ],
  modelConfigs: {}
};

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    this.datasetSprite = new Image();
    const imgRequest = new Promise((resolve, reject) => {
      this.datasetSprite.crossOrigin = '';
      this.datasetSprite.onload = () => {
        this.datasetSprite.width = this.datasetSprite.naturalWidth;
        this.datasetSprite.height = this.datasetSprite.naturalHeight;
        resolve();
      };
      this.datasetSprite.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const requestPromise = fetch(MNIST_LABELS_PATH, {mode: 'arraybuffer'});
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, requestPromise]);

    this.labels = new Uint8Array(await labelsResponse.arrayBuffer());
    // return dl.tensor2d(buffer, [buffer.length / NUM_CLASSES, NUM_CLASSES]);

    this.dataset = new tf.XhrDataset(mnistConfig);
    await this.dataset.fetchData();

    this.dataset.normalizeWithinBounds(0, -1, 1);
    this.trainingData = this.getTrainingData();
    this.testData = this.getTestData();

    this.trainIndices =
        tf.util.createShuffledIndices(this.trainingData[0].length);
    this.testIndices = tf.util.createShuffledIndices(this.testData[0].length);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(batchSize, this.trainingData, () => {
      this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
      return this.trainIndices[this.shuffledTrainIndex];
    });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, this.testData, () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    let xs = null;
    let labels = null;

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const x = data[0][idx].reshape([1, 784]);
      xs = concatWithNulls(xs, x);

      const label = data[1][idx].reshape([1, 10]);
      labels = concatWithNulls(labels, label);
    }
    return {xs, labels};
  }

  getTrainingData() {
    const [images, labels] = this.dataset.getData();

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
  }

  getTestData() {
    const data = this.dataset.getData();
    if (data == null) {
      return null;
    }
    const [images, labels] = this.dataset.getData();

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
  }
}

/**
 * TODO(nsthorat): Add math.stack, similar to np.stack, which will avoid the
 * need for us allowing concating with null values.
 */
function concatWithNulls(x1, x2) {
  if (x1 == null && x2 == null) {
    return null;
  }
  if (x1 == null) {
    return x2;
  } else if (x2 === null) {
    return x1;
  }
  const axis = 0;
  return x1.concat(x2, axis);
}
