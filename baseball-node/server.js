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
const pitch_type = require('./pitch_type');
const sleep = require('./utils').sleep;

const TIMEOUT_BETWEEN_EPOCHS_MS = 500;
const PORT = 8001;

async function run() {
  const port = process.env.PORT || PORT;
  const server = http.createServer();
  const io = socketio(this.server);

  server.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`);
  });

  io.emit('accuracyPerClass', await pitch_type.evaluate());
  // socket.sendAccuracyPerClass(await pitchModel.evaluate());
  await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);

  while (true) {
    // Fit one epoch...

    //     -    await pitchModel.train(1, progress =>
    //     socket.sendProgress(progress));
    // -    socket.sendAccuracyPerClass(
    // -        await pitchModel.evaluate(socket.useTrainingData));

    // io.emit('progress', progress);
    // this.io.emit('accuracyPerClass', accPerClass);

    await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);
  }
}

run();
