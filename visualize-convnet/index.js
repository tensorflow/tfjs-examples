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

const vizSection = document.getElementById('viz-section');
const vizTypeSelect = document.getElementById('viz-type');

function normalizePath(path) {
  return path.startsWith('dist/') ? path.slice(5) : path;
}

async function run() {
  // Empty the viz section.
  while (vizSection.firstChild) {
    vizSection.removeChild(vizSection.firstChild);
  }

  const vizType = vizTypeSelect.value;

  const activationManifest =
      await (await fetch('activation/activation-manifest.json')).json();
  const filterManifest =
      await (await fetch('filters/filters-manifest.json')).json();
  console.log(activationManifest);

  if (vizType === 'activation') {
    const layerNames = Object.keys(activationManifest.layerName2FilePaths);
    layerNames.sort();
    for (let i = 0; i < layerNames.length; ++i) {
      const layerName = layerNames[i];
      const filePaths = activationManifest.layerName2FilePaths[layerName];

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
        const activationImg = document.createElement('img');
        activationImg.classList.add('activation-image');
        activationImg.src = normalizePath(filePaths[j]);
        filterDiv.appendChild(activationImg);
        layerFiltersDiv.appendChild(filterDiv);
      }
      layerDiv.appendChild(layerFiltersDiv);
      vizSection.appendChild(layerDiv);
    }
  } else if (vizType === 'filters') {
    for (let i = 0; i < filterManifest.layers.length; ++i) {
      const layerName = filterManifest.layers[i].layerName;
      const filePaths = filterManifest.layers[i].filePaths;

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
        if (vizType === 'filters') {
          const filterImg = document.createElement('img');
          filterImg.classList.add('filter-image');
          filterImg.src = normalizePath(filePaths[j]);
          filterDiv.appendChild(filterImg);
        }
        layerFiltersDiv.appendChild(filterDiv);
      }
      layerDiv.appendChild(layerFiltersDiv);
      vizSection.appendChild(layerDiv);
    }
  } else if (vizType === 'cam') {
    const imgDiv = document.createElement('div');
    imgDiv.classList.add('cam-img');
    const img = document.createElement('img');
    img.src = normalizePath(activationManifest.camImagePath);
    imgDiv.appendChild(img);

    const winnerDiv = document.createElement('div');
    winnerDiv.textContent = `${activationManifest.topClass} ` +
        `(p=${activationManifest.topProb.toFixed(4)})`;

    vizSection.appendChild(imgDiv);
    vizSection.appendChild(winnerDiv);
  }
};

vizTypeSelect.addEventListener('change', run);

run();
