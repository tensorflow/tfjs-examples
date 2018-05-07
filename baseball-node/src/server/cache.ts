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

// tslint:disable-next-line:max-line-length
import {isValidPitchTypeData} from 'baseball-pitchfx-data';
// tslint:disable-next-line:max-line-length
import {Pitch, pitchFromType} from 'baseball-pitchfx-types';
import * as uuid from 'uuid';

import {loadPitchData} from '../pitch-data';
import {PitchPredictionMessage, PitchPredictionUpdateMessage, PitchTypeModel} from '../pitch-type-model';

const PITCH_CLASSES = 7;
const MAX_PITCHES_CACHE = 10;

export class PitchCache {
  predictionMessages: PitchPredictionMessage[];
  predictionQueue: PitchPredictionMessage[];
  trainCache: Pitch[];

  constructor(private model: PitchTypeModel) {
    this.predictionMessages = [];
    this.predictionQueue = [];
    this.trainCache = [];
  }

  loadTestData(): void {
    const testPitches = loadPitchData('dist/pitch_type_test_data.json');

    // Select 1 pitch each from the dataset.
    const range = testPitches.length / PITCH_CLASSES;
    for (let i = 0; i < PITCH_CLASSES; i++) {
      this.cachePitch(testPitches[range * i]);
    }
  }

  cachePitches(pitches: Pitch[]): void {
    pitches.forEach((pitch) => {
      if (isValidPitchTypeData(pitch)) {
        this.trainCache.unshift(pitch);
      }
    });
    const pitchData = pitches.length > MAX_PITCHES_CACHE ?
        pitches.slice(pitches.length - MAX_PITCHES_CACHE) :
        pitches;
    pitchData.forEach(pitch => this.cachePitch(pitch));
  }

  cachePitch(pitch: Pitch): void {
    const message = {
      uuid: uuid.v4(),
      pitch,
      actual: pitchFromType(pitch.pitch_code),
      pitch_classes: this.model.predict(pitch),
      class_percentage: this.model.pitchTypeClassAverage(pitch.pitch_code)
    };

    this.predictionMessages.unshift(message);
    this.predictionQueue.unshift(message);

    if (this.predictionMessages.length > MAX_PITCHES_CACHE) {
      this.predictionMessages.pop();
    }
    if (this.predictionQueue.length > MAX_PITCHES_CACHE) {
      this.predictionQueue.pop();
    }
  }

  clear(): void {
    this.predictionMessages = [];
    this.predictionQueue = [];
  }

  clearQueue(): void {
    this.predictionQueue = [];
  }

  clearTrainCache(): void {
    this.trainCache = [];
  }

  size(): number {
    return this.predictionMessages.length;
  }

  queueSize(): number {
    return this.predictionQueue.length;
  }

  trainSize(): number {
    return this.trainCache.length;
  }

  generateUpdatedPredictions(): PitchPredictionUpdateMessage[] {
    const updates = [] as PitchPredictionUpdateMessage[];
    for (let i = 0; i < this.predictionMessages.length; i++) {
      const message = this.predictionMessages[i];
      updates.push({
        uuid: message.uuid,
        pitch_classes: this.model.predict(message.pitch),
        class_percentage:
            this.model.pitchTypeClassAverage(message.pitch.pitch_code)
      });
    }
    return updates;
  }
}
