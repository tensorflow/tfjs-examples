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
import {Pitch, PitchKeys} from 'baseball-pitchfx-types';
import {readFileSync} from 'fs';

/**
 * Map of training fields for a Pitch with a min/max range for data
 * normalization.
 */
export type PitchTrainFields = {
  key: PitchKeys,
  min?: number,
  max?: number
};

/**
 * Callback function for returning the label for a given Pitch.
 */
export type GenerateLabel = (pitch: Pitch) => number;

/**
 * Definition of batch of pitches for training. Contains Training data and
 * labels as Tensors.
 */
export type PitchDataBatch = {
  labels: tf.Tensor2D; pitches: tf.Tensor2D;
};

/**
 * Converts a Pitch object to a Tensor based on the given training fields.
 */
export function createPitchTensor(
    pitch: Pitch, fields: PitchTrainFields[]): tf.Tensor2D {
  const shape = [1, fields.length];
  const values = pitchTrainDataArray(pitch, fields);
  return tf.tensor2d(new Float32Array(values), shape as [number, number]);
}

/**
 * Loads a JSON training file and the content to a Pitch array.
 */
export function loadPitchData(filename: string): Pitch[] {
  const pitches = [] as Pitch[];
  const content = readFileSync(filename, 'utf-8').split('\n');
  for (let i = 0; i < content.length - 1; i++) {
    pitches.push(JSON.parse(content[i]) as Pitch);
  }
  return pitches;
}

/**
 * Data class that enables easy of converting Pitch objects into training
 * Tensors.
 */
export class PitchData {
  batchSize: number;
  fields: PitchTrainFields[];
  batches: PitchDataBatch[];
  generateLabel: GenerateLabel;
  labelCount: number;
  index: number;

  constructor(
      filename: string, batchSize: number, fields: PitchTrainFields[],
      labelCount: number, generateLabel: GenerateLabel) {
    this.batchSize = batchSize;
    this.fields = fields;
    this.generateLabel = generateLabel;
    this.labelCount = labelCount;
    this.index = 0;

    // Load and convert training data to batches.
    this.batches = [] as PitchDataBatch[];
    const pitchData = loadPitchData(filename);
    tf.util.shuffle(pitchData);
    this.batches = this.generateBatch(pitchData);
  }

  /**
   * Generates a batch of training data for a list of Pitch objects.
   */
  generateBatch(pitches: Pitch[]): PitchDataBatch[] {
    const batches = [] as PitchDataBatch[];
    let index = 0;
    while (index < pitches.length) {
      batches.push(
          this.singlePitchBatch(pitches.slice(index, index + this.batchSize)));

      index += this.batchSize;
      if (pitches.length - index < this.batchSize) {
        this.batchSize = pitches.length - index;
      }
    }
    return batches;
  }

  private singlePitchBatch(pitches: Pitch[]): PitchDataBatch {
    const shape = [pitches.length, this.fields.length];
    const data = new Float32Array(tf.util.sizeFromShape(shape));
    const labels = [] as number[];

    return tf.tidy(() => {
      let offset = 0;
      for (let i = 0; i < pitches.length; i++) {
        const pitch = pitches[i];

        // Assign pitch fields
        data.set(pitchTrainDataArray(pitch, this.fields), offset);
        offset += this.fields.length;

        // Assign label
        labels.push(this.generateLabel(pitch));
      }
      return {
        pitches: tf.tensor2d(data, shape as [number, number]),
        labels:
            tf.oneHot(tf.tensor1d(labels, 'int32'), this.labelCount).toFloat()
      };
    });
  }

  /**
   * Returns entire list of stored pitch training batches.
   */
  pitchBatches(): PitchDataBatch[] {
    return this.batches;
  }
}

function pitchTrainDataArray(
    pitch: Pitch, fields: PitchTrainFields[]): number[] {
  const values = [];
  for (let i = 0; i < fields.length; i++) {
    const field = fields[i];
    values.push(normalize(pitch[field.key] as number, field.min, field.max));
  }
  return values;
}

function normalize(value: number, min: number, max: number): number {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value - min) / (max - min);
}
