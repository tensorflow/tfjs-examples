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

import {antirectifier} from './custom_layer';

function customLayerDemo() {
  let imgElement = document.getElementById('cat');
  // Layer expects first dimension to be batch, therefore expandDims.
  const img = tf.browser.fromPixels(imgElement).toFloat().expandDims(0);
  const layer = antirectifier();
  const [posTensor, negTensor] = tf.split(layer.apply(img), 2, 3);
  const posCanvas = document.createElement('canvas');
  tensorToCanvas(posTensor, posCanvas);
  document.getElementById('output_image_1').appendChild(posCanvas);
  const negCanvas = document.createElement('canvas');
  tensorToCanvas(negTensor, negCanvas);
  document.getElementById('output_image_2').appendChild(negCanvas);
}

function tensorToCanvas(tensor, canvas) {
  const ctx = canvas.getContext('2d');
  const [batch, height, width, nChan] = tensor.shape;
  console.assert(nChan == 3);
  console.assert(batch == 1);
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(width, height);
  const data = tensor.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const i4 = i * 4;
    const i3 = i * 3;
    imageData.data[i4 + 0] = data[i3 + 0] * 2;
    imageData.data[i4 + 1] = data[i3 + 1] * 2;
    imageData.data[i4 + 2] = data[i3 + 2] * 2;
    imageData.data[i4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
};

customLayerDemo();
