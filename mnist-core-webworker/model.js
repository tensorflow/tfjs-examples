/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {expose} from 'comlink';
import {MnistData as Data} from './data';

// Hyperparameters.
const LEARNING_RATE = 0.1;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 200;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;

class Model {
  data = new Data();

  optimizer = tf.train.sgd(LEARNING_RATE);

  // Variables that we want to optimize
  conv1OutputDepth = 8;
  conv1Weights =
      tf.variable(tf.randomNormal([5, 5, 1, this.conv1OutputDepth], 0, 0.1));

  conv2InputDepth = this.conv1OutputDepth;
  conv2OutputDepth = 16;
  conv2Weights = tf.variable(tf.randomNormal(
      [5, 5, this.conv2InputDepth, this.conv2OutputDepth], 0, 0.1));

  fullyConnectedWeights = tf.variable(tf.randomNormal(
      [7 * 7 * this.conv2OutputDepth, LABELS_SIZE], 0,
      1 / Math.sqrt(7 * 7 * this.conv2OutputDepth)));
  fullyConnectedBias = tf.variable(tf.zeros([LABELS_SIZE]));

  async load() {
    await this.data.load();
  }

  // Train the model.
  async train(log) {
    const returnCost = true;

    for (let i = 0; i < TRAIN_STEPS; i++) {
      const cost = this.optimizer.minimize(() => {
        const batch = this.data.nextTrainBatch(BATCH_SIZE);
        return this.loss(batch.labels, this.model(batch.xs));
      }, returnCost);

      log(`loss[${i}]: ${cost.dataSync()}`);

      // Note lack of nextFrame
      // await tf.nextFrame();
    }
  }

  // Predict the digit number from a batch of input images.
  predict(x) {
    const pred = tf.tidy(() => {
      const axis = 1;
      return this.model(x).argMax(axis);
    });
    return Array.from(pred.dataSync());
  }

  // Our actual model
  model(inputXs) {
    const xs = inputXs.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 1);

    const strides = 2;
    const pad = 0;

    // Conv 1
    const layer1 = tf.tidy(() => {
      return xs.conv2d(this.conv1Weights, 1, 'same')
          .relu()
          .maxPool([2, 2], strides, pad);
    });

    // Conv 2
    const layer2 = tf.tidy(() => {
      return layer1.conv2d(this.conv2Weights, 1, 'same')
          .relu()
          .maxPool([2, 2], strides, pad);
    });

    // Final layer
    return layer2.as2D(-1, this.fullyConnectedWeights.shape[0])
        .matMul(this.fullyConnectedWeights)
        .add(this.fullyConnectedBias);
  }

  // Loss function
  loss(labels, ys) {
    return tf.losses.softmaxCrossEntropy(labels, ys).mean();
  }

  // Given a logits or label vector, return the class indices.
  classesFromLabel(y) {
    const axis = 1;
    const pred = y.argMax(axis);

    return Array.from(pred.dataSync());
  }

  // Proxy for ui to call
  test(batchSize) {
    const batch = this.data.nextTestBatch(batchSize);
    const predictions = this.predict(batch.xs);
    const labels = this.classesFromLabel(batch.labels);

    const {xs} = batch;
    const imagesData = [];
    for (let i = 0; i < batchSize; i++) {
      const image = xs.slice([i, 0], [1, xs.shape[1]]);

      imagesData.push(image.flatten().dataSync());
    }

    return {imagesData, predictions, labels};
  }
}

expose(Model, self);
