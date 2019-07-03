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

import * as tf from '@tensorflow/tfjs';
import {ipcRenderer} from 'electron';

console.log('In renderer.js');  // DEBUG
console.log(tf);

ipcRenderer.on('get-files-or-directories-response', (event, arg) => {
  console.log('get-files-or-directories-response:', arg);  // DEBUG
  console.log(arg.xs);
  console.log(arg.ys);
  console.log(arg.ys.print());
});

const filesDialogButton = document.getElementById('files-dialog-button');
filesDialogButton.addEventListener('click', () => {
  ipcRenderer.send('get-files');
});

const directoriesDialogButton = document.getElementById('directories-dialog-button');
directoriesDialogButton.addEventListener('click', () => {
  ipcRenderer.send('get-directories');
});

const searchButton = document.getElementById('search-button');
searchButton.style.display = 'none';
