/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

require('@tensorflow/tfjs-node');

const http = require('http');
const socketio = require('socket.io');
const model = require('./pitch_type_model');
const sleep = require('./utils').sleep;

const TIMEOUT_BETWEEN_EPOCHS_MS = 500;
const PORT = 8001;

class Socket {
  constructor() {
    this.port = process.env.PORT || PORT;
    this.server = http.createServer();
    this.io = socketio(this.server);
    this.useTrainingData = false;
  }

  listen() {
    this.server.listen(this.port, () => {
      console.log(`  > Running socket on port: ${this.port}`);
    });
  }

  sendAccuracyPerClass(accPerClass) {
    this.io.emit('accuracyPerClass', accPerClass);
  }

  sendProgress(progress) {
    this.io.emit('progress', progress);
  }
}

async function run() {
  const socket = new Socket();
  socket.listen();
  // socket.sendAccuracyPerClass(await pitchModel.evaluate());
  await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);

  while (true) {
    // Fit one epoch...

    // Send accuracy

    await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);
  }
}

run();
