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
import {Pitch, PitchClass, pitchFromType} from 'baseball-pitchfx-types';

import {PitchModel} from './abstract-pitch-model';
// tslint:disable-next-line:max-line-length
import {createPitchTensor, PitchData} from './pitch-data';

// min/max constants from training data:
const VX0_MIN = -18.885;
const VX0_MAX = 18.065;
const VY0_MIN = -152.463;
const VY0_MAX = -86.374;
const VZ0_MIN = -15.5146078412997;
const VZ0_MAX = 9.974;
const AX_MIN = -48.0287647107959;
const AX_MAX = 30.592;
const AY_MIN = 9.397;
const AY_MAX = 49.18;
const AZ_MIN = -49.339;
const AZ_MAX = 2.95522851438373;
const START_SPEED_MIN = 59;
const START_SPEED_MAX = 104.4;

/**
 * Model to classify pitch types based on initial release acceleration,
 * velocity, speed, and pitcher hand (left or right).
 */
export class PitchTypeModel extends PitchModel {
  constructor() {
    super('pitch-type');

    this.fields = [
      {key: 'vx0', min: VX0_MIN, max: VX0_MAX},
      {key: 'vy0', min: VY0_MIN, max: VY0_MAX},
      {key: 'vz0', min: VZ0_MIN, max: VZ0_MAX},
      {key: 'ax', min: AX_MIN, max: AX_MAX},
      {key: 'ay', min: AY_MIN, max: AY_MAX},
      {key: 'az', min: AZ_MIN, max: AZ_MAX},
      {key: 'start_speed', min: START_SPEED_MIN, max: START_SPEED_MAX},
      {key: 'left_handed_pitcher'}
    ];

    this.data = new PitchData(
        'dist/pitch_type_training_data.json', 100, this.fields, 7,
        (pitch) => pitch.pitch_code);

    const model = tf.sequential();
    model.add(tf.layers.dense(
        {units: 250, activation: 'relu', inputShape: [this.fields.length]}));
    model.add(tf.layers.dense({units: 175, activation: 'relu'}));
    model.add(tf.layers.dense({units: 150, activation: 'relu'}));
    model.add(tf.layers.dense({units: 7, activation: 'softmax'}));
    model.compile({
      optimizer: tf.train.adam(),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    this.model = model;
  }

  /**
   * Returns sorted pitch type classification predictions for a given Pitch.
   */
  predict(pitch: Pitch): PitchClass[] {
    const pitchTensor = createPitchTensor(pitch, this.fields);
    const predict = this.model.predict(pitchTensor) as tf.Tensor;
    const values = predict.dataSync();

    let list = [] as PitchClass[];
    for (let i = 0; i < values.length; i++) {
      list.push({value: values[i], type: pitchFromType(i), pitch_code: i});
    }
    list = list.sort((a, b) => b.value - a.value);
    return list;
  }
}
