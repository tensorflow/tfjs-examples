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

import {createServer, Server} from 'http';
import * as socketio from 'socket.io';

import {PitchCache} from './cache';

const PORT = 8001;

export type SocketCallback = () => void;

export class Socket {
  server: Server;
  io: socketio.Server;
  port: string|number;

  constructor(
      private pitchCache: PitchCache,
      private toggleLiveDataCallback: SocketCallback) {
    this.port = process.env.PORT || PORT;
    this.server = createServer();
    this.io = socketio(this.server);
  }

  listen(): void {
    this.server.listen(this.port, () => {
      console.log(`  > Running socket on port: ${this.port}`);
    });

    this.io.on('connect', (socket) => {
      console.log(`  > Client connected on port: ${this.port} - sending ${
          this.pitchCache.size()} cached messages`);
      socket.emit('pitch_predictions', this.pitchCache.predictionMessages);

      socket.on('event', (data: string) => {
        if (data === 'toggle_live_data') {
          this.toggleLiveDataCallback();
        }
      });
    });
  }

  broadcastPredictions(): void {
    if (this.pitchCache.predictionQueue.length > 0) {
      console.log(`  > sending : ${this.pitchCache.queueSize()} new messages`);
      this.io.emit('pitch_predictions', this.pitchCache.predictionQueue);
      this.pitchCache.clearQueue();
    }
  }

  broadcastUpdatedPredictions(): void {
    const messages = this.pitchCache.generateUpdatedPredictions();
    console.log(`  > sending : ${messages.length} prediction updates`);
    this.io.emit('prediction_updates', messages);
  }
}
