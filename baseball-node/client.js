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


import io from 'socket.io-client';

const socket = io('localhost:8001');

// const socket = socketioClient(
//     SOCKET_URL, {reconnectionDelay: 300, reconnectionDelayMax: 300});

// socket.on('hi', (msg) => {
//   console.log(msg);
// });

console.log('yo', socket);
socket.on('connect', function() {
  console.log('connect()');
  // liveButton.style.display = 'block';
  // liveButton.textContent = 'Test Live';
});

socket.on('accuracyPerClass', (accPerClass) => {
  console.log('...');
  // plotAccuracyPerClass(accPerClass);
});

socket.on('disconnect', () => {
  console.log('disconnect()');
  // liveButton.style.display = 'block';
  // document.getElementById('waiting-msg').style.display = 'block';
  // document.getElementById('table').style.display = 'none';
});
