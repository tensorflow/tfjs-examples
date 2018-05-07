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

import {AccuracyPerClass, TrainProgress} from '../types';

const PORT = 8001;

export class Socket {
  server: Server;
  io: socketio.Server;
  port: string|number;

  constructor() {
    this.port = process.env.PORT || PORT;
    this.server = createServer();
    this.io = socketio(this.server);
  }

  listen(): void {
    this.server.listen(this.port, () => {
      console.log(`  > Running socket on port: ${this.port}`);
    });
  }

  sendAccuracyPerClass(accPerClass: AccuracyPerClass) {
    this.io.emit('accuracyPerClass', accPerClass);
  }

  sendProgress(progress: TrainProgress) {
    this.io.emit('progress', progress);
  }
}
