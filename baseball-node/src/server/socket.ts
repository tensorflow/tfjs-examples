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
import * as tf from '@tensorflow/tfjs';
// tslint:disable-next-line:max-line-length
import {Pitch, pitchFromType, PitchPredictionMessage, PitchPredictionUpdateMessage} from 'baseball-pitchfx-types';
import {createServer, Server} from 'http';
import * as socketio from 'socket.io';
import * as uuid from 'uuid';
import {loadPitchData} from '../pitch-data';
import {PitchTypeModel} from '../pitch-type-model';
import {getRandomInt} from '../utils';
import {TrainProgress} from '../abstract-pitch-model';

const PORT = 8001;
const PITCH_COUNT = 12;

export class Socket {
  server: Server;
  io: socketio.Server;
  port: string|number;

  pitches: Pitch[];
  pitchPredictionMessages: PitchPredictionMessage[];

  constructor(private pitchModel: PitchTypeModel) {
    this.port = process.env.PORT || PORT;
    this.server = createServer();
    this.io = socketio(this.server);

    this.pitchPredictionMessages = [];
    this.pitches = loadPitchData('dist/pitch_type_test_data.json');
    tf.util.shuffle(this.pitches);

    for (let i = 0; i < PITCH_COUNT; i++) {
      this.pitchPredictionMessages.push(this.generateResponse(
          this.pitches[getRandomInt(this.pitches.length)]));
    }
  }

  listen(): void {
    this.server.listen(this.port, () => {
      console.log(`  > Running socket on port: ${this.port}`);
    });

    this.io.on('connect', (socket) => {
      console.log(`  > Client connected on port: ${this.port} - sending ${
          this.pitchPredictionMessages.length} cached messages`);
      socket.emit('pitch_predictions', this.pitchPredictionMessages);
    });
  }

  broadcastUpdatedPredictions() {
    const updates = [] as PitchPredictionUpdateMessage[];
    const predictions = this.pitchPredictionMessages;
    predictions.forEach((prediction) => {
      updates.push(this.generatePredictionUpdateMessage(
          prediction.uuid, prediction.pitch));
    });
    if (updates.length > 0) {
      console.log(`  > sending : ${updates.length} prediction updates`);
      this.io.emit('prediction_updates', updates);
    }
  }

  sendProgress(progress: TrainProgress) {
    this.io.emit('progress', progress);
  }

  generateResponse(pitch: Pitch): PitchPredictionMessage {
    return {
      uuid: uuid.v4(),
      pitch,
      actual: pitchFromType(pitch.pitch_code),
      pitch_classes: this.pitchModel.predict(pitch),
      strike_zone_classes: []
    };
  }

  generatePredictionUpdateMessage(uuid: string, pitch: Pitch):
      PitchPredictionUpdateMessage {
    return {
      uuid,
      pitch_classes: this.pitchModel.predict(pitch),
      strike_zone_classes: []
    };
  }
}
