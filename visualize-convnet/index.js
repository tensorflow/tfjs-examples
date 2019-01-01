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
const vizTypeSelect = document.getElementById('viz-type');

async function run() {
  while (filtersSection.firstChild) {
    filtersSection.removeChild(filtersSection.firstChild);
  }

  const vizType = vizTypeSelect.value;

  const activationManifest = await (await fetch('activation/activation-manifest.json')).json();
  const filterManifest = await (await fetch('filters/filters-manifest.json')).json();

  for (let i = 0; i < filterManifest.layers.length; ++i) {
    const layerName = filterManifest.layers[i].layerName;
    const filePaths = filterManifest.layers[i].filePaths;
    const activationPaths = activationManifest.layerName2FilePaths[layerName];

    const layerDiv = document.createElement('div');
    layerDiv.classList.add('layer-div');
    const layerNameSpan = document.createElement('div');
    layerNameSpan.textContent = `Layer "${layerName}"`;
    layerDiv.appendChild(layerNameSpan);

    const layerFiltersDiv = document.createElement('div');
    layerFiltersDiv.classList.add('layer-filters');
    for (let j = 0; j < filePaths.length; ++j) {
      const filterDiv = document.createElement('div');
      filterDiv.classList.add('filter-div');
      if (vizType === 'filters' || vizType === 'both') {
        const filterImg = document.createElement('img');
        filterImg.classList.add('filter-image');
        filterImg.src = filePaths[j].startsWith('dist/') ?
            filePaths[j].slice(5) : filePaths[j];
        filterDiv.appendChild(filterImg);
      }
      if (vizType === 'activation' || vizType === 'both') {
        const activationImg = document.createElement('img');
        activationImg.classList.add('activation-image');
        activationImg.src = activationPaths[j].startsWith('dist/') ?
            activationPaths[j].slice(5) : activationPaths[j];
        console.log(activationImg.src);  // DEBUG
        filterDiv.appendChild(activationImg);
      }

      layerFiltersDiv.appendChild(filterDiv);
    }
    layerDiv.appendChild(layerFiltersDiv);
    
    filtersSection.appendChild(layerDiv);
  }
};

vizTypeSelect.addEventListener('change', run);

run();
