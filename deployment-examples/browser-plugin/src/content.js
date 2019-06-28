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

var imageMeta = {};

const setClickedImageTitles = () => {
  console.log("RUNNING setClickedImageTitles");
  const images = document.getElementsByTagName('img');
  const keys = Object.keys(imageMeta);
  for(u = 0; u < keys.length; u++) {
    const url = keys[u];
    const meta = imageMeta[url];
    for (i = 0; i < images.length; i++) {
      var img = images[i];
      if (img.src === meta.url) {
        img.title = "I WAS CLICKED ^_^";
        delete keys[u];
        delete imageMeta[url];
      }
    }
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log(`the tab was ${sender.tab}`);
  console.log(`the message was ${message}`);
  if (message && message.payload && message.action === 'IMAGE_CLICK_PROCESSED') {
    console.log("Heared IMAGE_CLICK_PROCESSED");
    const { payload } = message;
    if (payload && payload.url) {
      imageMeta[payload.url] = payload;
      setClickedImageTitles();
    }
  }
});

// const setImageTitles = () => {
//   const images = document.getElementsByTagName('img');
//   const keys = Object.keys(imageMeta);
//   for(u = 0; u < keys.length; u++) {
//     const url = keys[u];
//     const meta = imageMeta[url];
//     for (i = 0; i < images.length; i++) {
//       var img = images[i];
//       if (img.src === meta.url) {
//         img.title = img.src + `:\n\n${img.title}\n\n` + JSON.stringify(meta.predictions);
//         delete keys[u];
//         delete imageMeta[url];
//       }
//     }
//   }
// }




// chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
//   if (message && message.payload && message.action === 'IMAGE_PROCESSED') {
//     const { payload } = message;
//     if (payload && payload.url) {
//       imageMeta[payload.url] = payload;
//       setImageTitles();
//     }
//   }
// });



// window.addEventListener('load', setImageTitles, false);
// window.addEventListener('load', setClickedImageTitles, false);
