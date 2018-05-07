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
import {Pitch} from 'baseball-pitchfx-types';

import {PitchData, PitchDataBatch, PitchTrainFields} from './pitch-data';

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

  /**
   * Trains the model with a specific list of pitches.
   * @param pitches An array of Pitch objects for training.
   */
  async trainWithPitches(
      pitches: Pitch[], callback: (progress: TrainProgress) => void) {
    const batch = this.data.generateBatch(pitches);
    batch.forEach((b) => {
      this.trainInternal(b, callback, true);
    });
  }

  private async trainInternal(
      batch: PitchDataBatch, callback: (progress: TrainProgress) => void,
      log = false) {
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