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
import {Pitch, StrikeZoneClass} from 'baseball-pitchfx-types';

import {PitchModel} from './abstract-pitch-model';
// tslint:disable-next-line:max-line-length
import {createPitchTensor, PitchData} from './pitch-data';

// Constants from training data:
const PX_MIN = -2.65170604056843;
const PX_MAX = 2.842899614;
const PZ_MIN = -2.01705841594049;
const PZ_MAX = 6.06644249133382;
const SZ_TOP_MIN = 2.85;
const SZ_TOP_MAX = 4.241794863019148;
const SZ_BOT_MIN = 1.248894636863092;
const SZ_BOT_MAX = 2.2130980270561516;

/**
 * Model that learns to call balls and strikes based on ball placement at home
 * plate, strike zone height, and side the batter is hitting from.
 */
export class StrikeZoneModel extends PitchModel {
  constructor() {
    super('strike-zone');

    this.fields = [
      {key : 'px', min : PX_MIN, max : PX_MAX},
      {key : 'pz', min : PZ_MIN, max : PZ_MAX},
      {key : 'sz_top', min : SZ_TOP_MIN, max : SZ_TOP_MAX},
      {key : 'sz_bot', min : SZ_BOT_MIN, max : SZ_BOT_MAX},
      {key : 'left_handed_batter'}
    ];

    this.data =
        new PitchData('dist/strike_zone_training_data.json', 50, this.fields, 2,
                      (pitch) => pitch.type === 'S' ? 0 : 1);

    const model = tf.sequential();
    model.add(tf.layers.dense({
      units : 20,
      activation : 'relu',
      inputShape : [ this.fields.length ]
    }));
    model.add(tf.layers.dense({units : 10, activation : 'relu'}));
    model.add(tf.layers.dense({units : 2, activation : 'softmax'}));
    model.compile({
      optimizer : tf.train.adam(),
      loss : 'categoricalCrossentropy',
      metrics : [ 'accuracy' ]
    });
    this.model = model;
  }

  /**
   * Returns ball/strike prediction (sorted) for a given pitch.
   */
  predict(pitch: Pitch): StrikeZoneClass[] {
    const pitchTensor = createPitchTensor(pitch, this.fields);
    const predict = this.model.predict(pitchTensor) as tf.Tensor;
    const values = predict.dataSync();

    let list = [] as StrikeZoneClass[];
    list.push({value : values[0], strike : 1});
    list.push({value : values[1], strike : 0});
    list = list.sort((a, b) => b.value - a.value);
    return list;
  }
}
