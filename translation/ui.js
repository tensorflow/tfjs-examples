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

export function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

export function setEnglish(text, translate) {
  document.getElementById('englishSentence').value = text;
  document.getElementById('frenchSentence').textContent = translate(text);
}

export function setTranslationFunction(translate) {
  const englishElement = document.getElementById('englishSentence');
  englishElement.addEventListener('input', (e) => {
    const inputSentence = englishElement.value;
    document.getElementById('frenchSentence').textContent =
        translate(inputSentence);
  });
}

export function disableLoadModelButtons() {
  document.getElementById('load-pretrained-remote').disabled = true;
  document.getElementById('load-pretrained-local').disabled = true;
}
