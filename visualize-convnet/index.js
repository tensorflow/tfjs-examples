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

const filtersSection = document.getElementById('filters-section');

async function run() {
  const manifest = await (await fetch('filters/manifest.json')).json();
  for (let i = 0; i < manifest.layers.length; ++i) {
    const layerName = manifest.layers[i].layerName;
    const filePaths = manifest.layers[i].filePaths;

    const layerDiv = document.createElement('div');
    layerDiv.classList.add('layer-div');
    const layerNameSpan = document.createElement('div');
    layerNameSpan.textContent = `Layer "${layerName}"`;
    layerDiv.appendChild(layerNameSpan);

    const layerFiltersDiv = document.createElement('div');
    for (let j = 0; j < filePaths.length; ++j) {
      const img = document.createElement('img');
      img.classList.add('filter-image');
      const filePath = filePaths[j].startsWith('dist/') ?
          filePaths[j].slice(5) : filePaths[j];
      img.src = filePath;
      layerFiltersDiv.appendChild(img);
    }
    layerDiv.appendChild(layerFiltersDiv);
    
    filtersSection.appendChild(layerDiv);
  }
};

run();
