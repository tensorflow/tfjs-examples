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

import * as tf from '@tensorflow/tfjs';
import {PitchData, PitchDataBatch, PitchTrainFields} from './pitch-data';
import {pitchFromType} from 'baseball-pitchfx-types';
import {AccuracyPerClass} from './types';

/** Info about progress during training. */
export interface TrainProgress {
  accuracy: number;
  loss: number;
}

/**
 * Abstract base class for defining Pitch ML models.
 */
export abstract class PitchModel {
  protected fields: PitchTrainFields[];
  protected model: tf.Model;
  protected data: PitchData;
  protected totalTrainSteps: number;
  protected modelName: string;

  constructor(modelName: string) {
    this.modelName = modelName;
    this.totalTrainSteps = 0;
  }

  trainSteps(): number {
    return this.totalTrainSteps;
  }

  /**
   * Trains a model with saved training data for a given number of epochs.
   * @param epochs Number of passes through the training data.
   */
  async train(epochs: number, callback: (progress: TrainProgress) => void) {
    const batches = this.data.pitchBatches();
    for (let i = 0; i < epochs; i++) {
      for (let j = 0; j < batches.length; j++) {
        await this.trainInternal(batches[j], callback, j % 10 === 0);
      }
    }
  }

  /** Computes accuracy per class for the entire training set. */
  async evaluate(): Promise<AccuracyPerClass> {
    const batches = this.data.pitchBatches();
    const correctPerClass: number[] = [];
    const countPerClass: number[] = [];
    const numClasses = batches[0].labels.shape[1];
    for (let i = 0; i < numClasses; i++) {
      correctPerClass[i] = 0;
      countPerClass[i] = 0;
    }

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      const predictionBatch = this.model.predict(batch.pitches) as tf.Tensor;
      const labelIndicesBatch = batch.labels.argMax(1);
      const isCorrectBatch =
        await labelIndicesBatch.equal(predictionBatch.argMax(1)).data();
      const labelBatch = await labelIndicesBatch.data();
      for (let i = 0; i < isCorrectBatch.length; i++) {
        const labelIndex = labelBatch[i];
        const isCorrect = isCorrectBatch[i];
        countPerClass[labelIndex]++;
        if (isCorrect) {
          correctPerClass[labelIndex]++;
        }
      }
    }
    
    // Return a dict that maps a class name to accuracy.
    const result: AccuracyPerClass = {};
    correctPerClass.forEach((correct, i) => {
      result[pitchFromType(i)] = {training: correct / countPerClass[i]};
    });
    return result;
  }

  private async trainInternal(batch: PitchDataBatch,
      callback: (progress: TrainProgress) => void, log = false) {
    await this.model.fit(batch.pitches, batch.labels, {
      epochs: 1,
      shuffle: false,
      validationData: [batch.pitches, batch.labels],
      batchSize: 100,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (log) {
            callback({accuracy: logs.acc, loss: logs.loss});
            console.log(
                `[${this.modelName} : step ${this.totalTrainSteps}] loss: ${
                    logs.loss.toFixed(4)} accuracy: ${logs.acc.toFixed(4)}`);
          }
        }
      }
    });
    this.totalTrainSteps++;
  }
}