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
import {PitchPredictionMessage, PitchPredictionUpdateMessage} from 'baseball-pitchfx-types';
import {UniqueQueue} from 'containers.js';
import * as socketioClient from 'socket.io-client';
import Vue from 'vue';

import Pitch from './Pitch.vue';

const maxPitches = 10;

const SOCKET = 'http://localhost:8001/';

const data = {
  predictionsQueue: new UniqueQueue<PitchPredictionMessage>(maxPitches),
  predictions: [] as PitchPredictionMessage[],
  predictionMap: new Map<string, number>(),
  connected: false,
  socket: socketioClient
};

// tslint:disable-next-line:no-default-export
export default Vue.extend({
  data() {
    return data;
  },
  components: {Pitch},
  // tslint:disable-next-line:object-literal-shorthand
  mounted: function() {
    this.$data.socket = socketioClient(SOCKET);

    this.$data.socket.on(
        'pitch_predictions', (data: PitchPredictionMessage[]) => {
          console.log(`Found :${data.length} pitches`);
          data.forEach((prediction) => {
            this.predictionsQueue.push(prediction);
          });
          this.predictions = this.predictionsQueue.values();
          this.predictionMap.clear();
          for (let i = 0; i < this.predictions.length; i++) {
            this.predictionMap.set(this.predictions[i].uuid, i);
          }
        });

    this.$data.socket.on(
        'prediction_updates', (data: PitchPredictionUpdateMessage[]) => {
          data.forEach((update) => {
            const index = this.predictionMap.get(update.uuid);
            if (index !== undefined) {
              this.predictions[index].pitch_classes = update.pitch_classes;
              this.predictions[index].strike_zone_classes =
                  update.strike_zone_classes;
            }
          });
        });

    this.$data.socket.on('disconnect', () => {
      this.$data.connected = false;
      this.predictionMap.clear();
      this.predictions = [];
      this.predictionsQueue.clear();
    });

    this.$data.socket.on('connect', () => {
      this.$data.connected = true;
    });
  },

  methods: {
    loadLive: function() {
      this.$data.predictionMap.clear();
      this.$data.predictions = [];
      this.$data.predictionsQueue.clear();
      this.$data.socket.emit('event', 'toggle_live_data');
    }
  }
});
