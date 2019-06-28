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
const TEXT_DIV_CLASSNAME = 'tfjs_mobilenet_extension_text';

// Sets the image title for all images in imageMeta
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

const addTextOverImage = () => {
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

// Produces a short text string summarizing the prediction
// Input prediction should be a list of {className: string, prediction: float}
// objects.
function textContentFromPrediction(predictions) {
  if (!predictions || predictions.length < 1) {
    return "No prediction 🙁";
  }
  // Confident.
  if (predictions[0].probability >= 0.5) {
    return "😄 " + predictions[0].className + "!";
  }
  // Not Confident.
  if (predictions[0].probability >= 0.1 && predictions[0].probability < 0.5) {
    return predictions[0].className + "?... \n Maybe " + predictions[1].className;
  }
  // Very not confident.
  if (predictions[0].probability < 0.1) {
    return "😕  " + predictions[0].className + 
    "????... \n Maybe " + predictions[1].className + "????";
  }
}

// Returns a list of all DOM image elements pointing to the same image url.
function getImageElementsWithUrl(srcUrl) {
  console.log(`filtering for ${srcUrl}`);
  console.log(`found ${document.getElementsByTagName('img').length} total images`);
  const imgElArr = Array.from(document.getElementsByTagName('img'));
  // const srcArr = imgElArr.map(x => x.src);
  const filtImgElArr = imgElArr.filter(x => x.src === srcUrl);  
  // const filtSrcArr = filtImgElArr.map(x => x.src);
  // console.log(srcArr);
  // console.log(filtSrcArr);
  // console.log(filtImgElArr);
  return filtImgElArr;
  // return Array.prototype.filter.call(document.getElementsByTagName('img'),
  //    (function (a) {a.src === srcUrl}));
}

// Removes all previous text predictions.
function removeTextElements() {
  const textDivs = document.getElementsByClassName(TEXT_DIV_CLASSNAME);
  for (const div of textDivs) {
    div.parentNode.removeChild(div);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log(`the tab was ${sender.tab}`);
  console.log(`the message was ${JSON.stringify(message)}`);
  if (message && message.payload && message.action === 'IMAGE_CLICK_PROCESSED') {
    console.log("Heared IMAGE_CLICK_PROCESSED");
    const { payload } = message;
    if (payload && payload.url) {
      imageMeta[payload.url] = payload;
      setClickedImageTitles();
      // There may be more than one image with the same url... 
      const imgElements = getImageElementsWithUrl(payload.url);
      console.log(imgElements);
      if (imgElements.length === 0) {
        console.log(`Could not find an image with url ${payload.url}`)
      }
      for (const imgNode of imgElements) {
        console.log('do dom manip here');
        const imgParent = imgNode.parentElement;
        const divNode = document.createElement('div');
        divNode.style.position = 'relative';
        divNode.style.textAlign = 'center';
        divNode.style.colore = 'white';
        const textDiv = document.createElement('div');
        textDiv.textContent = textContentFromPrediction(payload.predictions); 
        textDiv.className = 'tfjs_mobilenet_extension_text';
        textDiv.style.position = 'absolute';
        textDiv.style.top = '50%';
        textDiv.style.left = '50%';
        textDiv.style.transform = 'translate(-50%, -50%)';
        textDiv.style.fontSize = '34px';
        textDiv.style.fontFamily = 'Google Sans,sans-serif';
        textDiv.style.fontWeight = '700';
        textDiv.style.color = 'white';
        textDiv.style.lineHeight = '1em';
        textDiv.style["-webkit-text-fill-color"] = "white";
        textDiv.style["-webkit-text-stroke-width"] = "1px";
        textDiv.style["-webkit-text-stroke-color"] =  "black";
        // Add the divNode as a peer, right next to the imageNode.
        imgParent.insertBefore(divNode, imgNode);
        // Move the imageNode to inside the divNode;
        divNode.appendChild(imgNode);
        // Add the text node right after the image node;
        divNode.appendChild(textDiv);
      }
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

// Clear previous predictions if the user clicks the left mouse button.
window.addEventListener('click', clickHandler, false);
function clickHandler(mouseEvent) {
  if (mouseEvent.button == 0) {
    removeTextElements();
  }
}